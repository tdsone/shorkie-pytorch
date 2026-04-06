import json

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextvars import ContextVar
from dataclasses import dataclass


DNA_CHANNELS = 4  # num of channels when one-hot encoding DNA
NUM_SPECIES = 165  # number of species in the pretrained language model
SPECIES_OFFSET = (
    5  # channels 0-3 = nucleotides, channel 4 = reserved, channels 5+ = species
)
R64_SPECIES_INDEX = 109  # S. cerevisiae (R64) index in the species encoding


@dataclass(frozen=True)
class BuildConfig:
    bn_momentum: float


_current_config: ContextVar[BuildConfig] = ContextVar("_current_config")

# ──────────────────────────────────────────────────────────────
# Convolution blocks  (channels-first: B, C, T)
# ──────────────────────────────────────────────────────────────


class ConvDNA(nn.Module):
    """Initial convolution on one-hot DNA.

    Accepts the full species-encoded input (170 channels by default):
    channels 0-3 = one-hot nucleotides, channel 4 = reserved,
    channels 5-169 = one-hot species indicator.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        norm_type,
        activation,
        in_channels=SPECIES_OFFSET + NUM_SPECIES,  # 170
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            out_channels=filters,
            bias=True,
            padding="same",
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ConvNAC(nn.Module):
    """Norm → Activation → Conv1d  (channels-first)."""

    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        cfg = _current_config.get()
        self.norm = nn.BatchNorm1d(
            num_features=in_channels,
            momentum=1 - cfg.bn_momentum,  # torch momentum = 1 - keras momentum
            eps=0.001,  # match TF default
        )
        self.gelu = nn.GELU(approximate="tanh")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gelu(x)
        x = self.conv(x)
        return x


class ResTowerBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_convs, dropout, pool_size, kernel_size
    ) -> None:
        super().__init__()
        self.conv0 = ConvNAC(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.conv_stack = nn.Sequential(
            *[
                ConvNAC(
                    in_channels=out_channels, out_channels=out_channels, kernel_size=1
                )
                for _ in range(1, num_convs)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size, stride=pool_size, ceil_mode=True
        )
        self.res_scale = nn.Parameter(torch.zeros(out_channels, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_init = self.conv0(x)
        residual = self.conv_stack(x_init)
        residual_scaled = self.res_scale * self.dropout(residual)
        pre_pool = x_init + residual_scaled  # representation saved before pooling
        x = self.pool(pre_pool)
        return x, pre_pool


class ResTower(nn.Module):
    def __init__(
        self,
        filters_init,
        filters_end,
        divisible_by,
        kernel_size,
        num_convs,
        dropout,
        pool_size,
        repeat,
    ) -> None:
        super().__init__()

        FILTERS_MULT = np.exp(np.log(filters_end / filters_init) / (repeat - 1))

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        _blocks: list[nn.Module] = []
        rep_filters = filters_init
        in_channels = filters_init
        for _ in range(repeat):
            out_channels = _round(rep_filters)
            _blocks.append(
                ResTowerBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_convs=num_convs,
                    dropout=dropout,
                    pool_size=pool_size,
                    kernel_size=kernel_size,
                )
            )
            rep_filters *= FILTERS_MULT
            in_channels = out_channels

        self.blocks = nn.ModuleList(_blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        reprs = []
        for block in self.blocks:
            x, pre_pool = block(x)
            reprs.append(pre_pool)
        return x, reprs


# ──────────────────────────────────────────────────────────────
# Transformer blocks  (channels-last: B, T, C)
# ──────────────────────────────────────────────────────────────


def positional_features_central_mask(positions, feature_size, seq_length):
    """Central-mask positional features (matching TF baskerville)."""
    pow_rate = np.exp(np.log(seq_length + 1) / feature_size).astype("float32")
    center_widths = pow_rate ** torch.arange(
        1, feature_size + 1, dtype=torch.float32, device=positions.device
    )
    center_widths = center_widths - 1  # (feature_size,)
    return (center_widths > torch.abs(positions).unsqueeze(-1)).float()


def positional_features(positions, feature_size, seq_length, symmetric=False):
    """Relative positional encodings."""
    num_components = 1 if symmetric else 2
    num_basis_per_class = feature_size // num_components

    embeddings = positional_features_central_mask(
        positions, num_basis_per_class, seq_length
    )

    if not symmetric:
        embeddings = torch.cat(
            [embeddings, torch.sign(positions).unsqueeze(-1) * embeddings], dim=-1
        )
    return embeddings


def relative_shift(x):
    """Shift relative logits (TransformerXL-style)."""
    B, H, T1, T2 = x.shape
    # Prepend a column of zeros
    x = torch.cat([torch.zeros(B, H, T1, 1, device=x.device, dtype=x.dtype), x], dim=-1)
    x = x.reshape(B, H, T2 + 1, T1)
    x = x[:, :, 1:, :]  # remove first row
    x = x.reshape(B, H, T1, T2)
    x = x[:, :, :, : (T2 + 1) // 2]
    return x


class MultiheadAttention(nn.Module):
    """Multi-head attention with relative positional encoding (matching baskerville)."""

    def __init__(
        self,
        value_size,
        key_size,
        heads,
        num_position_features,
        attention_dropout_rate=0.05,
        positional_dropout_rate=0.01,
        content_position_bias=True,
        zero_initialize=True,
    ):
        super().__init__()
        self._value_size = value_size
        self._key_size = key_size
        self._num_heads = heads
        self._num_position_features = num_position_features
        self._attention_dropout_rate = attention_dropout_rate
        self._positional_dropout_rate = positional_dropout_rate
        self._content_position_bias = content_position_bias

        key_proj_size = key_size * heads
        embedding_size = value_size * heads

        self.q_layer = nn.Linear(embedding_size, key_proj_size, bias=False)
        self.k_layer = nn.Linear(embedding_size, key_proj_size, bias=False)
        self.v_layer = nn.Linear(embedding_size, embedding_size, bias=False)

        self.r_k_layer = nn.Linear(num_position_features, key_proj_size, bias=False)
        self.r_w_bias = nn.Parameter(torch.zeros(1, heads, 1, key_size))
        self.r_r_bias = nn.Parameter(torch.zeros(1, heads, 1, key_size))

        self.embedding_layer = nn.Linear(embedding_size, embedding_size)
        if zero_initialize:
            nn.init.zeros_(self.embedding_layer.weight)
            nn.init.zeros_(self.embedding_layer.bias)

    def _to_multihead(self, x, channels_per_head):
        """(B, T, H*C) -> (B, H, T, C)"""
        B, T, _ = x.shape
        x = x.reshape(B, T, self._num_heads, channels_per_head)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        B, T, _ = x.shape

        q = self._to_multihead(self.q_layer(x), self._key_size)  # (B,H,T,K)
        k = self._to_multihead(self.k_layer(x), self._key_size)  # (B,H,T,K)
        v = self._to_multihead(self.v_layer(x), self._value_size)  # (B,H,T,V)

        q = q * (self._key_size**-0.5)

        # Content logits
        content_logits = torch.matmul(
            q + self.r_w_bias, k.transpose(-2, -1)
        )  # (B,H,T,T)

        # Relative position logits
        distances = torch.arange(
            -T + 1, T, dtype=torch.float32, device=x.device
        ).unsqueeze(0)
        pos_enc = positional_features(
            distances, self._num_position_features, T, symmetric=False
        )  # (1, 2T-1, num_pos_features)

        if self.training:
            pos_enc = F.dropout(pos_enc, p=self._positional_dropout_rate)

        r_k = self._to_multihead(
            self.r_k_layer(pos_enc), self._key_size
        )  # (1,H,2T-1,K)

        if self._content_position_bias:
            relative_logits = torch.matmul(q + self.r_r_bias, r_k.transpose(-2, -1))
        else:
            relative_logits = torch.matmul(self.r_r_bias, r_k.transpose(-2, -1))
            relative_logits = relative_logits.expand(B, -1, T, -1)

        relative_logits = relative_shift(relative_logits)
        logits = content_logits + relative_logits

        weights = F.softmax(logits, dim=-1)
        if self.training:
            weights = F.dropout(weights, p=self._attention_dropout_rate)

        output = torch.matmul(weights, v)  # (B,H,T,V)
        output = output.permute(0, 2, 1, 3)  # (B,T,H,V)
        output = output.reshape(B, T, -1)  # (B,T,H*V)

        return self.embedding_layer(output)


class TransformerBlock(nn.Module):
    """Single transformer block: MHA + FFN, both with pre-norm and residual."""

    def __init__(
        self,
        channels,
        key_size,
        heads,
        num_position_features,
        dropout,
        dense_expansion=2.0,
    ):
        super().__init__()
        value_size = channels // heads

        # Attention sub-block
        self.attn_norm = nn.LayerNorm(channels, eps=0.001)  # match TF default
        self.mha = MultiheadAttention(
            value_size=value_size,
            key_size=key_size,
            heads=heads,
            num_position_features=num_position_features,
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # FFN sub-block
        self.ffn_norm = nn.LayerNorm(channels, eps=0.001)  # match TF default
        expansion_filters = int(dense_expansion * channels)
        self.ffn_dense1 = nn.Linear(channels, expansion_filters)
        self.ffn_dense2 = nn.Linear(expansion_filters, channels)
        self.ffn_dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # MHA with residual
        h = self.attn_norm(x)
        h = self.mha(h)
        h = self.attn_dropout(h)
        x = x + h

        # FFN with residual
        h = self.ffn_norm(x)
        h = self.ffn_dense1(h)
        h = self.ffn_dropout1(h)
        h = F.relu(h)
        h = self.ffn_dense2(h)
        h = self.ffn_dropout2(h)
        x = x + h

        return x


class TransformerTower(nn.Module):
    def __init__(
        self,
        key_size,
        heads,
        num_position_features,
        dropout,
        repeat,
        # consumed but not used in this reduced impl
        mha_l2_scale=0,
        l2_scale=0,
        kernel_initializer="he_normal",
        **kwargs,
    ) -> None:
        super().__init__()
        # channels will be set lazily on first forward, or explicitly via _set_channels
        self._key_size = key_size
        self._heads = heads
        self._num_position_features = num_position_features
        self._dropout = dropout
        self._repeat = repeat
        self.blocks: nn.ModuleList | None = None

    def _set_channels(self, channels: int):
        if self.blocks is not None:
            return
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels=channels,
                    key_size=self._key_size,
                    heads=self._heads,
                    num_position_features=self._num_position_features,
                    dropout=self._dropout,
                )
                for _ in range(self._repeat)
            ]
        )

    def forward(self, x: torch.Tensor):
        if self.blocks is None:
            self._set_channels(x.shape[-1])
        for block in self.blocks:
            x = block(x)
        return x


# ──────────────────────────────────────────────────────────────
# U-Net upsampling  (channels-last: B, T, C)
# ──────────────────────────────────────────────────────────────


class SeparableConv1d(nn.Module):
    """Depthwise separable conv1d (channels-last I/O)."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding="same", groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x: (B, T, C) -> permute for conv -> back
        x = x.permute(0, 2, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x.permute(0, 2, 1)


class UNetConv(nn.Module):
    """U-Net upsampling block: upsample current, add skip, separable conv."""

    def __init__(
        self,
        channels: int,
        skip_channels: int,
        kernel_size: int,
        upsample_conv: bool,
        bn_momentum: float,
    ) -> None:
        super().__init__()
        self.upsample_conv = upsample_conv

        # Norm + activation on main path
        self.norm1 = nn.BatchNorm1d(channels, momentum=1 - bn_momentum, eps=0.001)
        # Norm + activation on skip path
        self.norm2 = nn.BatchNorm1d(skip_channels, momentum=1 - bn_momentum, eps=0.001)

        self.gelu = nn.GELU(approximate="tanh")

        if upsample_conv:
            self.dense1 = nn.Linear(channels, channels)
        self.dense2 = nn.Linear(skip_channels, channels)

        self.sep_conv = SeparableConv1d(channels, kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), skip: (B, 2T, C_skip) — both channels-last

        # Normalize (BN needs channels-first)
        current = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        skip_out = self.norm2(skip.permute(0, 2, 1)).permute(0, 2, 1)

        # Activate
        current = self.gelu(current)
        skip_out = self.gelu(skip_out)

        # Dense projections
        if self.upsample_conv:
            current = self.dense1(current)
        skip_out = self.dense2(skip_out)

        # Upsample current 2× via nearest-neighbor
        current = current.permute(0, 2, 1)  # (B,C,T) for interpolate
        current = F.interpolate(current, scale_factor=2, mode="nearest")
        current = current.permute(0, 2, 1)  # back to (B,T*2,C)

        # Add
        current = current + skip_out

        # Separable conv
        current = self.sep_conv(current)

        return current


# ──────────────────────────────────────────────────────────────
# Cropping
# ──────────────────────────────────────────────────────────────


class Cropping1D(nn.Module):
    def __init__(self, cropping) -> None:
        super().__init__()
        self.cropping = cropping

    def forward(self, x):
        # x: (B, T, C) channels-last
        if self.cropping > 0:
            return x[:, self.cropping : -self.cropping, :]
        return x


# ──────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────


class Shorkie(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        token = _current_config.set(BuildConfig(bn_momentum=config["bn_momentum"]))
        try:
            self._build(config)
        finally:
            _current_config.reset(token)

    def _build(self, config: dict):
        trunk_cfg = config["trunk"]

        # ── Build conv_dna ──
        conv_dna_cfg = dict(trunk_cfg[0])
        del conv_dna_cfg["name"]
        self.conv_dna = ConvDNA(**conv_dna_cfg)

        # ── Build res_tower ──
        rt_cfg = dict(trunk_cfg[1])
        del rt_cfg["name"]
        self.res_tower = ResTower(**rt_cfg)

        # Compute filter sizes per res_tower block (needed for UNet skip dims)
        rt = rt_cfg
        fmult = np.exp(
            np.log(rt["filters_end"] / rt["filters_init"]) / (rt["repeat"] - 1)
        )
        _round = lambda x: int(np.round(x / rt["divisible_by"]) * rt["divisible_by"])
        self._skip_channels = []
        rep_f = rt["filters_init"]
        for _ in range(rt["repeat"]):
            self._skip_channels.append(_round(rep_f))
            rep_f *= fmult

        # Output channels of res_tower
        out_channels = self._skip_channels[-1]

        # ── Build transformer_tower ──
        tt_cfg = dict(trunk_cfg[2])
        del tt_cfg["name"]
        self.transformer_tower = TransformerTower(**tt_cfg)
        self.transformer_tower._set_channels(out_channels)

        # ── Build UNet convs ──
        # Each UNet block uses a skip from progressively earlier res_tower blocks
        # UNet 0 uses skip from block[-1] (384ch, 256 positions)
        # UNet 1 uses skip from block[-2] (320ch, 512 positions)
        # UNet 2 uses skip from block[-3] (256ch, 1024 positions)
        unet_cfgs = [c for c in trunk_cfg if c["name"] == "unet_conv"]
        self.unet_convs = nn.ModuleList()
        for i, uc in enumerate(unet_cfgs):
            skip_ch = self._skip_channels[-(i + 1)]
            self.unet_convs.append(
                UNetConv(
                    channels=out_channels,
                    skip_channels=skip_ch,
                    kernel_size=uc["kernel_size"],
                    upsample_conv=uc["upsample_conv"],
                    bn_momentum=config["bn_momentum"],
                )
            )

        # ── Build Cropping1D ──
        crop_cfg = [c for c in trunk_cfg if c["name"] == "Cropping1D"][0]
        self.cropping = Cropping1D(crop_cfg["cropping"])

        # ── Final activation (applied after trunk, before head) ──
        self.final_gelu = nn.GELU(approximate="tanh")

        # ── Head ──
        head_cfg = config["head"]
        self.head = nn.Linear(out_channels, head_cfg["units"])

    def _expand_species_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Expand (B, 4, T) one-hot DNA to (B, 170, T) with species encoding.

        Matches baskerville's dna_1hot_mask_species_encoding:
        - Channels 0-3: nucleotide one-hot (copied from input)
        - Channel 4: reserved (zeros)
        - Channels 5-169: one-hot species indicator (channel 5+R64_SPECIES_INDEX=114 set to 1)
        """
        B, C, T = x.shape
        total_channels = SPECIES_OFFSET + NUM_SPECIES  # 170
        x_expanded = x.new_zeros(B, total_channels, T)
        x_expanded[:, :DNA_CHANNELS, :] = x
        x_expanded[:, SPECIES_OFFSET + R64_SPECIES_INDEX, :] = 1.0
        return x_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, T)  channels-first one-hot DNA

        # Expand to species-encoded input: (B, 4, T) -> (B, 170, T)
        x = self._expand_species_encoding(x)

        # Conv DNA
        x = self.conv_dna(x)  # (B, C, T)

        # Res tower — saves pre-pool reprs (channels-first)
        x, reprs = self.res_tower(x)  # x: (B, C, T'), reprs: list of (B, C_i, T_i)

        # Switch to channels-last for transformer + rest
        x = x.permute(0, 2, 1)  # (B, T', C)

        # Transformer tower
        x = self.transformer_tower(x)  # (B, T', C)

        # UNet upsampling with skip connections
        for i, unet in enumerate(self.unet_convs):
            # Skip from res_tower (channels-first), convert to channels-last
            skip = reprs[-(i + 1)].permute(0, 2, 1)
            x = unet(x, skip)

        # Cropping
        x = self.cropping(x)

        # Final activation + head
        x = self.final_gelu(x)
        x = self.head(x)
        x = F.softplus(x)

        return x  # (B, T_out, 5215)

    @staticmethod
    def from_tf_checkpoint(config: dict, h5_path: str) -> "Shorkie":
        """Load a Shorkie model with weights from an original TF/Keras H5 checkpoint.

        The checkpoint must contain the full 170-channel conv_dna kernel
        (4 nucleotide + 1 reserved + 165 species channels).
        """
        model = Shorkie(config)
        _load_tf_weights(model, h5_path)
        return model


# ──────────────────────────────────────────────────────────────
# Weight conversion: TF H5 → PyTorch
# ──────────────────────────────────────────────────────────────


def _get_tf_layer(h5root, name):
    """Get the parameter dict for a named TF layer from a Keras H5 checkpoint."""
    root = h5root["model_weights"] if "model_weights" in h5root else h5root
    grp = root[name]

    # Keras nests: layer_name/layer_name/... ; extracted: layer_name/param:0
    if name in grp and isinstance(grp[name], h5py.Group):
        grp = grp[name]  # unwrap keras double-nesting

    params = {}
    for key in grp.keys():
        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            params[key] = np.array(obj)
        elif isinstance(obj, h5py.Group):
            for subkey in obj.keys():
                params[f"{key}/{subkey}"] = np.array(obj[subkey])
    return params


def _load_tf_weights(model: Shorkie, h5_path: str):
    """Convert and load TF/Keras weights into the PyTorch model."""
    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f

        # Helper indices
        conv_i = 0
        bn_i = 0
        scale_i = 0
        ln_i = 0
        dense_i = 0
        mha_i = 0
        sep_i = 0

        def _conv_name(i):
            return "conv1d" if i == 0 else f"conv1d_{i}"

        def _bn_name(i):
            return "batch_normalization" if i == 0 else f"batch_normalization_{i}"

        def _scale_name(i):
            return "scale" if i == 0 else f"scale_{i}"

        def _ln_name(i):
            return "layer_normalization" if i == 0 else f"layer_normalization_{i}"

        def _dense_name(i):
            return "dense" if i == 0 else f"dense_{i}"

        def _mha_name(i):
            return "multihead_attention" if i == 0 else f"multihead_attention_{i}"

        def _sep_name(i):
            return "separable_conv1d" if i == 0 else f"separable_conv1d_{i}"

        # ── conv_dna ──
        p = _get_tf_layer(root, _conv_name(conv_i))
        conv_i += 1
        tf_kernel = p["kernel:0"]  # (K, C_in, C_out) in TF
        expected_in = SPECIES_OFFSET + NUM_SPECIES  # 170
        if tf_kernel.shape[1] != expected_in:
            raise ValueError(
                f"conv_dna kernel has {tf_kernel.shape[1]} input channels "
                f"(expected {expected_in}). Use the original TF checkpoint, not extracted weights."
            )
        model.conv_dna.conv.weight.data = torch.from_numpy(
            np.transpose(tf_kernel, (2, 1, 0))
        )
        model.conv_dna.conv.bias.data = torch.from_numpy(p["bias:0"])

        # ── res_tower ──
        for block in model.res_tower.blocks:
            # conv0: BN -> GELU -> Conv
            _load_bn(block.conv0.norm, _get_tf_layer(root, _bn_name(bn_i)))
            bn_i += 1
            _load_conv1d(block.conv0.conv, _get_tf_layer(root, _conv_name(conv_i)))
            conv_i += 1

            # conv_stack (1 layer for num_convs=2)
            for conv_nac in block.conv_stack:
                _load_bn(conv_nac.norm, _get_tf_layer(root, _bn_name(bn_i)))
                bn_i += 1
                _load_conv1d(conv_nac.conv, _get_tf_layer(root, _conv_name(conv_i)))
                conv_i += 1

            # Scale
            sp = _get_tf_layer(root, _scale_name(scale_i))
            scale_i += 1
            block.res_scale.data = torch.from_numpy(sp["scale:0"]).unsqueeze(
                1
            )  # (C,) -> (C,1)

        # ── transformer_tower ──
        for tblock in model.transformer_tower.blocks:
            # Attention LN
            _load_ln(tblock.attn_norm, _get_tf_layer(root, _ln_name(ln_i)))
            ln_i += 1

            # MHA
            mp = _get_tf_layer(root, _mha_name(mha_i))
            mha_i += 1
            mha = tblock.mha
            mha.q_layer.weight.data = torch.from_numpy(mp["q_layer/kernel:0"].T)
            mha.k_layer.weight.data = torch.from_numpy(mp["k_layer/kernel:0"].T)
            mha.v_layer.weight.data = torch.from_numpy(mp["v_layer/kernel:0"].T)
            mha.r_k_layer.weight.data = torch.from_numpy(mp["r_k_layer/kernel:0"].T)
            mha.r_w_bias.data = torch.from_numpy(mp["r_w_bias:0"])
            mha.r_r_bias.data = torch.from_numpy(mp["r_r_bias:0"])
            mha.embedding_layer.weight.data = torch.from_numpy(
                mp["embedding_layer/kernel:0"].T
            )
            mha.embedding_layer.bias.data = torch.from_numpy(
                mp["embedding_layer/bias:0"]
            )

            # FFN LN
            _load_ln(tblock.ffn_norm, _get_tf_layer(root, _ln_name(ln_i)))
            ln_i += 1

            # FFN Dense 1 (expansion)
            _load_linear(tblock.ffn_dense1, _get_tf_layer(root, _dense_name(dense_i)))
            dense_i += 1

            # FFN Dense 2 (projection)
            _load_linear(tblock.ffn_dense2, _get_tf_layer(root, _dense_name(dense_i)))
            dense_i += 1

        # ── UNet convs ──
        for unet in model.unet_convs:
            # BN on main path
            _load_bn(unet.norm1, _get_tf_layer(root, _bn_name(bn_i)))
            bn_i += 1
            # BN on skip path
            _load_bn(unet.norm2, _get_tf_layer(root, _bn_name(bn_i)))
            bn_i += 1

            # Dense on main path (if upsample_conv)
            if unet.upsample_conv:
                _load_linear(unet.dense1, _get_tf_layer(root, _dense_name(dense_i)))
                dense_i += 1

            # Dense on skip path
            _load_linear(unet.dense2, _get_tf_layer(root, _dense_name(dense_i)))
            dense_i += 1

            # Separable Conv1D
            sp = _get_tf_layer(root, _sep_name(sep_i))
            sep_i += 1
            # depthwise: TF (K, C, 1) -> PyTorch (C, 1, K)
            unet.sep_conv.depthwise.weight.data = torch.from_numpy(
                np.transpose(sp["depthwise_kernel:0"], (1, 2, 0))
            )
            # pointwise: TF (1, C_in, C_out) -> PyTorch (C_out, C_in, 1)
            unet.sep_conv.pointwise.weight.data = torch.from_numpy(
                np.transpose(sp["pointwise_kernel:0"], (2, 1, 0))
            )
            unet.sep_conv.pointwise.bias.data = torch.from_numpy(sp["bias:0"])

        # ── Head (final dense) ──
        _load_linear(model.head, _get_tf_layer(root, _dense_name(dense_i)))
        dense_i += 1


def _load_conv1d(conv: nn.Conv1d, params: dict):
    """Load TF Conv1D weights into PyTorch Conv1d."""
    # TF kernel: (K, C_in, C_out) -> PyTorch: (C_out, C_in, K)
    conv.weight.data = torch.from_numpy(np.transpose(params["kernel:0"], (2, 1, 0)))
    conv.bias.data = torch.from_numpy(params["bias:0"])


def _load_bn(bn: nn.BatchNorm1d, params: dict):
    """Load TF BatchNormalization weights into PyTorch BatchNorm1d."""
    bn.weight.data = torch.from_numpy(params["gamma:0"])
    bn.bias.data = torch.from_numpy(params["beta:0"])
    bn.running_mean.data = torch.from_numpy(params["moving_mean:0"])
    bn.running_var.data = torch.from_numpy(params["moving_variance:0"])


def _load_ln(ln: nn.LayerNorm, params: dict):
    """Load TF LayerNormalization weights into PyTorch LayerNorm."""
    ln.weight.data = torch.from_numpy(params["gamma:0"])
    ln.bias.data = torch.from_numpy(params["beta:0"])


def _load_linear(linear: nn.Linear, params: dict):
    """Load TF Dense weights into PyTorch Linear."""
    # TF: (in, out) -> PyTorch: (out, in)
    linear.weight.data = torch.from_numpy(params["kernel:0"].T)
    linear.bias.data = torch.from_numpy(params["bias:0"])


if __name__ == "__main__":
    with open("data/shorkie_params.json") as f:
        config = json.load(f)

    model = Shorkie(config["model"])
    print(model)

    # Test forward pass
    x = torch.randn(2, 4, 16384)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
