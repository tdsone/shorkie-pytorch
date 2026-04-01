import json
import torch
import torch.nn as nn
from abc import ABC
import numpy as np

DNA_CHANNELS = 4  # num of channels when one-hot encoding DNA

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:
    bn_momentum: float


_current_config: ContextVar[BuildConfig] = ContextVar("_current_config")


class SeqNNBlock(ABC):
    @classmethod
    def build_module(cls, config: dict) -> "SeqNNBlock":
        del config["name"]
        return cls(**config)


class ConvDNA(nn.Module, SeqNNBlock):
    """
    Reduced implementation of https://github.com/calico/baskerville-yeast/blob/88e89f48e7df73c0856ce93ae35e8878794d19e9/src/baskerville/blocks.py#L132
    """

    def __init__(
        self,
        filters,
        kernel_size,
        norm_type,  # norm_type is null in params.json which means no normalisation
        activation,  # activation is explicitly set to linear (=identity) overriding the global gelu
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=DNA_CHANNELS,
            kernel_size=kernel_size,
            out_channels=filters,
            bias=True,
            padding="same",
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class ConvNAC(nn.Module, SeqNNBlock):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        cfg = _current_config.get()
        self.norm = nn.BatchNorm1d(
            num_features=in_channels, momentum=1 - cfg.bn_momentum
        )
        self.gelu = nn.GELU()
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


class ResTowerBlock(nn.Module, SeqNNBlock):
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
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
                for j in range(1, num_convs)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(
            kernel_size=pool_size, stride=pool_size, ceil_mode=True
        )
        self.res_scale = nn.Parameter(torch.zeros(out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_init = self.conv0(x)
        residual = self.conv_stack(x_init)
        residual_scaled = self.res_scale * self.dropout(
            residual
        )  # channel wise scaling
        x = self.pool(x_init + residual_scaled)
        return x


class ResTower(nn.Module, SeqNNBlock):
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
        self.num_convs = num_convs

        FILTERS_MULT = np.exp(  # computes how we have to change the number of filters from block to block
            np.log(filters_end / filters_init) / (repeat - 1)
        )

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

            rep_filters *= FILTERS_MULT  # Increase number of filters

            in_channels = out_channels

        self.blocks: nn.Module = nn.Sequential(*_blocks)

    def forward(self, x: torch.Tensor):
        x = self.blocks(x)
        return x


class TransformerTower(nn.Module, SeqNNBlock):
    def __init__(
        self,
        key_size,
        heads,
        num_position_features,
        dropout,
        mha_l2_scale,
        l2_scale,
        kernel_initializer,
        repeat,
        **kwargs,
    ) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class UNetConv(nn.Module, SeqNNBlock):
    def __init__(self, kernel_size: int, upsample_conv: bool, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return


class Cropping1D(nn.Module, SeqNNBlock):
    def __init__(self, cropping) -> None:
        super().__init__()

    def forward(self, x):
        return x


class Shorkie(nn.Module):
    module_library: dict[str, nn.Module] = {
        "conv_dna": ConvDNA,
        "res_tower": ResTower,
        "transformer_tower": TransformerTower,
        "unet_conv": UNetConv,
        "Cropping1D": Cropping1D,
    }

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.trunk = self._build_trunk()

    def _build_trunk(self) -> nn.Module:
        token = _current_config.set(BuildConfig(bn_momentum=self.config["bn_momentum"]))

        try:
            blocks = [
                Shorkie.module_library[module_cfg["name"]].build_module(module_cfg)
                for module_cfg in self.config["trunk"]
            ]
        finally:
            _current_config.reset(token)

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.trunk(x)
        return x


if __name__ == "__main__":
    with open("data/shorkie_params.json") as f:
        config = json.load(f)

    print(config["model"]["trunk"])

    model = Shorkie(config["model"])

    print(model)
