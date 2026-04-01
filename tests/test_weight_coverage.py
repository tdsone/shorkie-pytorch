"""Report which TF checkpoint weights are loaded into the PyTorch model and which are not.

Requires:
    data/checkpoint_f0.h5  — Original TF checkpoint
"""

import json
import os
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import shorkie as shorkie_mod
from shorkie import Shorkie

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _checkpoint_available():
    return os.path.exists(os.path.join(DATA_DIR, "checkpoint_f0.h5"))


def _params_path():
    for name in ["params_gcs.json", "shorkie_params.json"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No params JSON found in data/")


@pytest.mark.skipif(not _checkpoint_available(), reason="No checkpoint_f0.h5 found")
def test_weight_coverage():
    """Report which TF checkpoint weights are used vs unused by the PyTorch model."""

    h5_path = os.path.join(DATA_DIR, "checkpoint_f0.h5")

    # Step 1: Enumerate all weight tensors in the checkpoint
    all_tf_weights = {}
    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"]

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_tf_weights[name] = obj.shape

        root.visititems(visitor)

    # Step 2: Track which weight tensors are accessed during loading.
    # We wrap _get_tf_layer to record which layer groups are accessed,
    # then map back to the individual weight tensor paths.
    accessed_layers = []
    original_get_tf_layer = shorkie_mod._get_tf_layer

    def tracking_get_tf_layer(h5root, name):
        accessed_layers.append(name)
        return original_get_tf_layer(h5root, name)

    with open(_params_path()) as f:
        config = json.load(f)

    with patch.object(shorkie_mod, "_get_tf_layer", side_effect=tracking_get_tf_layer):
        model = Shorkie.from_tf_checkpoint(config["model"], h5_path)

    # Step 3: Map accessed layer names to individual weight tensor paths.
    # TF checkpoint structure: layer_name/layer_name/param:0
    loaded_weights = set()
    for layer_name in accessed_layers:
        for weight_path in all_tf_weights:
            # Weight paths look like "conv1d/conv1d/kernel:0" — the layer name is the first component
            if weight_path.startswith(layer_name + "/"):
                loaded_weights.add(weight_path)

    unused_weights = set(all_tf_weights.keys()) - loaded_weights

    # Step 4: Report
    print(f"\n{'='*70}")
    print(f"WEIGHT COVERAGE REPORT")
    print(f"{'='*70}")
    print(f"Total TF weight tensors:  {len(all_tf_weights)}")
    print(f"Loaded into PyTorch:      {len(loaded_weights)}")
    print(f"Not loaded (unused):      {len(unused_weights)}")
    print(f"Coverage:                 {len(loaded_weights)/len(all_tf_weights)*100:.1f}%")

    if unused_weights:
        print(f"\n--- Unused weights ({len(unused_weights)}) ---")
        for w in sorted(unused_weights):
            print(f"  {w}: {all_tf_weights[w]}")

    print(f"\n--- Loaded weights ({len(loaded_weights)}) ---")
    for w in sorted(loaded_weights):
        print(f"  {w}: {all_tf_weights[w]}")

    # Step 5: Also check the reverse — PyTorch parameters without a TF source.
    # Count PyTorch parameters and compare.
    pt_params = {name: tuple(p.shape) for name, p in model.named_parameters()}
    pt_buffers = {name: tuple(b.shape) for name, b in model.named_buffers()}
    all_pt = {**pt_params, **pt_buffers}

    print(f"\n--- PyTorch model ---")
    print(f"Parameters:  {len(pt_params)}")
    print(f"Buffers:     {len(pt_buffers)} (batch norm running stats)")
    print(f"Total:       {len(all_pt)}")
    print(f"{'='*70}")


@pytest.mark.skipif(not _checkpoint_available(), reason="No checkpoint_f0.h5 found")
def test_parameter_count():
    """Compare the number of scalar parameters between the TF checkpoint and PyTorch model."""

    h5_path = os.path.join(DATA_DIR, "checkpoint_f0.h5")

    # Count TF parameters (scalar values) per weight tensor
    tf_counts = {}
    tf_total = 0
    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"]

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                layer_name = name.split("/")[0]
                n = int(np.prod(obj.shape))
                tf_counts[name] = n

        root.visititems(visitor)
        tf_total = sum(tf_counts.values())

    # Build PyTorch model and count
    with open(_params_path()) as fp:
        config = json.load(fp)
    model = Shorkie.from_tf_checkpoint(config["model"], h5_path)

    pt_param_counts = {name: int(p.numel()) for name, p in model.named_parameters()}
    pt_buffer_counts = {name: int(b.numel()) for name, b in model.named_buffers()}

    pt_param_total = sum(pt_param_counts.values())
    pt_buffer_total = sum(pt_buffer_counts.values())
    # Exclude num_batches_tracked buffers (PyTorch-only bookkeeping, not real weights)
    pt_buffer_counts_no_nbt = {
        k: v for k, v in pt_buffer_counts.items() if "num_batches_tracked" not in k
    }
    pt_buffer_total_no_nbt = sum(pt_buffer_counts_no_nbt.values())
    pt_total = pt_param_total + pt_buffer_total_no_nbt

    # Group TF counts by layer type for readability
    tf_by_type = {}
    for name, count in tf_counts.items():
        layer = name.split("/")[0]
        # Strip trailing _N to get the type
        parts = layer.rsplit("_", 1)
        if parts[-1].isdigit():
            layer_type = parts[0]
        else:
            layer_type = layer
        tf_by_type.setdefault(layer_type, 0)
        tf_by_type[layer_type] += count

    # Group PT counts by module type
    pt_by_type = {}
    for name, count in {**pt_param_counts, **pt_buffer_counts_no_nbt}.items():
        module = name.split(".")[0]
        pt_by_type.setdefault(module, 0)
        pt_by_type[module] += count

    # Report
    print(f"\n{'='*70}")
    print(f"PARAMETER COUNT COMPARISON")
    print(f"{'='*70}")

    print(f"\n--- TF checkpoint: {tf_total:,} total scalars ---")
    for layer_type in sorted(tf_by_type, key=lambda k: -tf_by_type[k]):
        print(f"  {layer_type:40s} {tf_by_type[layer_type]:>10,}")

    print(f"\n--- PyTorch model: {pt_total:,} total scalars ---")
    print(f"  (parameters: {pt_param_total:,}, buffers: {pt_buffer_total_no_nbt:,})")
    for module in sorted(pt_by_type, key=lambda k: -pt_by_type[k]):
        print(f"  {module:40s} {pt_by_type[module]:>10,}")

    diff = pt_total - tf_total
    print(f"\n--- Summary ---")
    print(f"  TF total:      {tf_total:>10,}")
    print(f"  PyTorch total:  {pt_total:>10,}")
    print(f"  Difference:     {diff:>+10,}")
    if diff == 0:
        print(f"  MATCH")
    else:
        print(f"  MISMATCH — investigate which parameters differ")
    print(f"{'='*70}")

    assert tf_total == pt_total, (
        f"Parameter count mismatch: TF has {tf_total:,}, PyTorch has {pt_total:,} "
        f"(diff {diff:+,})"
    )
