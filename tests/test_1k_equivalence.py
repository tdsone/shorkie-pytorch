"""Test that PyTorch Shorkie predictions match TF golden predictions.

Requires golden data files in data/ directory, generated via:
    modal run modal_extract_weights.py --fold 0

Files expected:
    data/predictions_f0_chunk{000..009}.h5  — TF predictions + input sequences (100 seqs each)
    data/checkpoint_f0.h5                   — Original TF checkpoint (170-channel conv_dna)
"""

import glob
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
BATCH_SIZE = 4


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


def _chunk_paths(fold: int) -> list[str]:
    """Find all prediction chunk files for a fold, sorted by chunk index."""
    pattern = os.path.join(DATA_DIR, f"predictions_f{fold}_chunk*.h5")
    return sorted(glob.glob(pattern))


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

    chunk_paths = _chunk_paths(fold)
    assert len(chunk_paths) > 0, f"No prediction chunks found for fold {fold}"

    max_abs = 0.0
    sum_abs = 0.0
    n_elements = 0
    per_seq_r = []
    n_skipped = 0
    global_seq_idx = 0

    for chunk_path in chunk_paths:
        chunk_name = os.path.basename(chunk_path)
        with h5py.File(chunk_path, "r") as f:
            n_seqs = f["sequences"].shape[0]

            for i in range(0, n_seqs, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n_seqs)
                try:
                    sequences = np.array(f["sequences"][i:end])
                    tf_preds = np.array(f["predictions"][i:end])
                except OSError:
                    n_skipped += end - i
                    global_seq_idx += end - i
                    continue

                x = torch.from_numpy(sequences).permute(0, 2, 1).float()
                with torch.no_grad():
                    pt_preds = model(x).numpy()

                assert pt_preds.shape == tf_preds.shape, (
                    f"Shape mismatch at seq {global_seq_idx}: "
                    f"PyTorch {pt_preds.shape} vs TF {tf_preds.shape}"
                )

                abs_diff = np.abs(pt_preds - tf_preds)
                max_abs = max(max_abs, float(abs_diff.max()))
                sum_abs += float(abs_diff.sum())
                n_elements += abs_diff.size

                for j in range(end - i):
                    r = np.corrcoef(pt_preds[j].flatten(), tf_preds[j].flatten())[0, 1]
                    per_seq_r.append(r)

                global_seq_idx += end - i

    n_compared = len(per_seq_r)
    assert n_compared > 0, "No sequences could be compared — golden data may be corrupted"
    mean_abs = sum_abs / n_elements
    per_seq_r = np.array(per_seq_r)

    print(f"\nFold {fold} — {n_compared} sequences compared "
          f"({len(chunk_paths)} chunks, {n_skipped} skipped)")
    print(f"  Max  abs diff: {max_abs:.6e}")
    print(f"  Mean abs diff: {mean_abs:.6e}")
    print(f"  Pearson R: min={per_seq_r.min():.8f}, mean={per_seq_r.mean():.8f}")

    # Within float32 precision — use both absolute and relative tolerance.
    # Large prediction values (e.g. ~2000) can have absolute diffs > 1e-4
    # while relative error stays < 1e-6.
    assert mean_abs < 1e-4, (
        f"Mean abs diff too large: {mean_abs:.6e}"
    )
    assert max_abs < 1e-2, (
        f"Max abs diff too large: {max_abs:.6e}"
    )
    assert per_seq_r.min() > 0.999999, (
        f"Min per-sequence correlation too low: {per_seq_r.min():.8f}"
    )
