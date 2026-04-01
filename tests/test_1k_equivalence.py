"""Test that PyTorch Shorkie predictions match TF golden predictions.

Requires golden data files in data/ directory, generated via:
    modal run modal_extract_weights.py --fold 0

Files expected:
    data/predictions_f0_chunk000.h5  — TF predictions + input sequences (100 seqs)
    data/checkpoint_f0.h5            — Original TF checkpoint (170-channel conv_dna)
"""

import json
import os
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shorkie import Shorkie

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BATCH_SIZE = 8


def _fold_data_available(fold: int) -> bool:
    weights = os.path.join(DATA_DIR, f"checkpoint_f{fold}.h5")
    chunk = os.path.join(DATA_DIR, f"predictions_f{fold}_chunk000.h5")
    return os.path.exists(weights) and os.path.exists(chunk)


def _available_folds():
    folds = [f for f in range(8) if _fold_data_available(f)]
    if not folds:
        return [pytest.param(-1, marks=pytest.mark.skip(
            reason="No golden data found. Run: modal run modal_extract_weights.py --fold 0"
        ))]
    return folds


def _params_path():
    for name in ["params_gcs.json", "shorkie_params.json"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No params JSON found in data/")


@pytest.fixture(scope="module")
def config():
    with open(_params_path()) as f:
        return json.load(f)


@pytest.mark.parametrize("fold", _available_folds())
def test_1k_equivalence(fold: int, config):
    """Compare PyTorch predictions against TF golden predictions for a specific fold."""

    # Load model from original TF checkpoint (170-channel conv_dna)
    weights_path = os.path.join(DATA_DIR, f"checkpoint_f{fold}.h5")
    model = Shorkie.from_tf_checkpoint(config["model"], weights_path)
    model.eval()

    # Load golden chunk
    chunk_path = os.path.join(DATA_DIR, f"predictions_f{fold}_chunk000.h5")
    with h5py.File(chunk_path, "r") as f:
        sequences = np.array(f["sequences"])     # (N, T, 4) channels-last
        tf_predictions = np.array(f["predictions"])  # (N, T_out, targets)

    n_seqs = len(sequences)
    all_predictions = []

    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = sequences[i : i + BATCH_SIZE]
            x = torch.from_numpy(batch).permute(0, 2, 1).float()
            pred = model(x)
            all_predictions.append(pred.numpy())

    pt_predictions = np.concatenate(all_predictions, axis=0)

    assert pt_predictions.shape == tf_predictions.shape, (
        f"Shape mismatch: PyTorch {pt_predictions.shape} vs TF {tf_predictions.shape}"
    )

    # Metrics
    abs_diff = np.abs(pt_predictions - tf_predictions)
    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()

    per_seq_r = []
    for i in range(n_seqs):
        r = np.corrcoef(pt_predictions[i].flatten(), tf_predictions[i].flatten())[0, 1]
        per_seq_r.append(r)
    per_seq_r = np.array(per_seq_r)

    print(f"\nFold {fold} — {n_seqs} sequences")
    print(f"  Max  abs diff: {max_abs:.6e}")
    print(f"  Mean abs diff: {mean_abs:.6e}")
    print(f"  Pearson R: min={per_seq_r.min():.8f}, mean={per_seq_r.mean():.8f}")

    # Within float32 precision
    assert np.allclose(pt_predictions, tf_predictions, atol=1e-4, rtol=1e-4), (
        f"Predictions not close enough. Max abs diff: {max_abs:.6e}"
    )
    assert per_seq_r.min() > 0.9999, (
        f"Min per-sequence correlation too low: {per_seq_r.min():.8f}"
    )
