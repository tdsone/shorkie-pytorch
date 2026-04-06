"""Compute equivalence statistics between TF and PyTorch Shorkie predictions.

Prints a summary table suitable for pasting into the README.

Usage:
    uv run python scripts/compute_equivalence.py [--fold 0]
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shorkie import Shorkie

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _chunk_paths(fold: int) -> list[str]:
    pattern = os.path.join(DATA_DIR, f"predictions_f{fold}_chunk*.h5")
    return sorted(glob.glob(pattern))


def _params_path():
    for name in ["params_gcs.json", "shorkie_params.json"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No params JSON found in data/")


class RunningPercentile:
    """Approximate percentiles using a fixed-size reservoir sample."""

    def __init__(self, reservoir_size=500_000):
        self.reservoir_size = reservoir_size
        self.reservoir = np.empty(reservoir_size, dtype=np.float32)
        self.n_filled = 0
        self.n_seen = 0

    def update(self, values: np.ndarray):
        flat = values.ravel()
        n_new = len(flat)

        if self.n_filled < self.reservoir_size:
            space = self.reservoir_size - self.n_filled
            take = min(space, n_new)
            self.reservoir[self.n_filled : self.n_filled + take] = flat[:take]
            self.n_filled += take
            flat = flat[take:]
            self.n_seen += take
            n_new = len(flat)
            if n_new == 0:
                return

        # Vectorized reservoir sampling for the rest
        self.n_seen += n_new
        # For each new element, probability of inclusion = reservoir_size / n_seen_so_far
        # Approximate: accept each with prob reservoir_size / n_seen
        probs = np.random.random(n_new)
        threshold = self.reservoir_size / self.n_seen
        mask = probs < threshold
        accepted = flat[mask]
        if len(accepted) > 0:
            indices = np.random.randint(0, self.reservoir_size, size=len(accepted))
            self.reservoir[indices] = accepted

    def percentile(self, q):
        return np.percentile(self.reservoir[: self.n_filled], q)

    def max(self):
        return self.reservoir[: self.n_filled].max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(_params_path()) as f:
        config = json.load(f)

    weights_path = os.path.join(DATA_DIR, f"checkpoint_f{args.fold}.h5")
    if not os.path.exists(weights_path):
        sys.exit(f"Checkpoint not found: {weights_path}")

    chunk_paths = _chunk_paths(args.fold)
    if not chunk_paths:
        sys.exit(f"No prediction chunks found for fold {args.fold}")

    print(f"Loading model (fold {args.fold})...")
    model = Shorkie.from_tf_checkpoint(config["model"], weights_path)
    model.to(device)
    model.eval()

    # Running statistics (no large accumulations)
    per_seq_r = []
    abs_sum = 0.0
    abs_max = 0.0
    rel_sum = 0.0
    n_elements = 0
    n_seqs = 0
    abs_pctl = RunningPercentile()
    rel_pctl = RunningPercentile()

    for chunk_path in chunk_paths:
        with h5py.File(chunk_path, "r") as f:
            n = f["sequences"].shape[0]
            for i in range(0, n, args.batch_size):
                end = min(i + args.batch_size, n)
                try:
                    sequences = np.array(f["sequences"][i:end])
                    tf_preds = np.array(f["predictions"][i:end])
                except OSError:
                    continue

                x = torch.from_numpy(sequences).permute(0, 2, 1).float().to(device)
                with torch.no_grad():
                    pt_preds = model(x).cpu().numpy()

                abs_diff = np.abs(pt_preds - tf_preds)
                denom = np.maximum(np.abs(tf_preds), 1e-8)
                rel_diff = abs_diff / denom

                abs_sum += abs_diff.sum()
                abs_max = max(abs_max, abs_diff.max())
                rel_sum += rel_diff.sum()
                n_elements += abs_diff.size
                abs_pctl.update(abs_diff)
                rel_pctl.update(rel_diff)

                for j in range(end - i):
                    r = np.corrcoef(pt_preds[j].flatten(), tf_preds[j].flatten())[0, 1]
                    per_seq_r.append(r)

                n_seqs += end - i
                print(f"\r  {n_seqs} sequences processed", end="", flush=True)

    print()

    per_seq_r = np.array(per_seq_r)

    print(f"\n{'='*60}")
    print(f"Equivalence: fold {args.fold}, {n_seqs} sequences")
    print(f"{'='*60}")

    print(f"\nPer-sequence Pearson R:")
    print(f"  min    = {per_seq_r.min():.8f}")
    print(f"  mean   = {per_seq_r.mean():.8f}")
    print(f"  median = {np.median(per_seq_r):.8f}")

    abs_mean = abs_sum / n_elements
    rel_mean = rel_sum / n_elements
    print(f"\nAbsolute error:")
    print(f"  mean   = {abs_mean:.2e}")
    print(f"  median = {abs_pctl.percentile(50):.2e}")
    print(f"  p95    = {abs_pctl.percentile(95):.2e}")
    print(f"  p99    = {abs_pctl.percentile(99):.2e}")
    print(f"  max    = {abs_max:.2e}")

    print(f"\nRelative error:")
    print(f"  mean   = {rel_mean:.2e}")
    print(f"  median = {rel_pctl.percentile(50):.2e}")
    print(f"  p95    = {rel_pctl.percentile(95):.2e}")
    print(f"  p99    = {rel_pctl.percentile(99):.2e}")
    print(f"  max    = {rel_pctl.max():.2e}")

    # README-ready table
    print(f"\n{'='*60}")
    print("Markdown table:")
    print(f"{'='*60}\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Sequences compared | {n_seqs} |")
    print(f"| Per-sequence Pearson R (mean) | {per_seq_r.mean():.8f} |")
    print(f"| Per-sequence Pearson R (min) | {per_seq_r.min():.8f} |")
    print(f"| Absolute error (mean) | {abs_mean:.2e} |")
    print(f"| Absolute error (median) | {abs_pctl.percentile(50):.2e} |")
    print(f"| Absolute error (p99) | {abs_pctl.percentile(99):.2e} |")
    print(f"| Absolute error (max) | {abs_max:.2e} |")
    print(f"| Relative error (mean) | {rel_mean:.2e} |")
    print(f"| Relative error (median) | {rel_pctl.percentile(50):.2e} |")
    print(f"| Relative error (p99) | {rel_pctl.percentile(99):.2e} |")


if __name__ == "__main__":
    main()
