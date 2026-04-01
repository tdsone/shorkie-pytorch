"""Run the full 1000-sequence equivalence test on Modal.

Loads the PyTorch model with extracted weights, runs inference on the same
1000 sequences used for the golden TF predictions, and compares.

Usage:
    modal run modal_test_equivalence.py --fold 0
"""

import modal

app = modal.App("shorkie-test-equivalence")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libhdf5-dev"])
    .pip_install(["torch", "h5py", "numpy"])
    .add_local_file("shorkie.py", remote_path="/root/shorkie.py")
)

output_volume = modal.Volume.from_name("shorkie-predictions")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/output": output_volume},
    timeout=3600,
    memory=16384,
)
def test_equivalence(fold: int):
    """Run full 1000-seq equivalence test on Modal."""
    import json
    import sys

    import h5py
    import numpy as np
    import torch

    # Upload shorkie.py as part of the mount — use inline import
    sys.path.insert(0, "/root")

    # ── Load model ──
    weights_path = f"/output/fold_{fold}/model_weights.h5"
    params_path = f"/output/fold_{fold}/model_weights.h5"  # params embedded? no.

    # We need the params. Let's read from the predictions file attrs or hardcode.
    # Actually we need to ship params. Let's just hardcode the path or embed.
    # The params are the same for all folds. Let's write them inline.
    import importlib.util
    spec = importlib.util.spec_from_file_location("shorkie", "/root/shorkie.py")
    shorkie_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shorkie_mod)
    Shorkie = shorkie_mod.Shorkie

    # Load config from embedded copy
    config = {
        "seq_length": 16384,
        "augment_rc": False,
        "augment_shift": 0,
        "activation": "gelu",
        "norm_type": "batch",
        "bn_momentum": 0.9,
        "kernel_initializer": "lecun_normal",
        "l2_scale": 1e-6,
        "trunk": [
            {"name": "conv_dna", "filters": 96, "kernel_size": 11, "norm_type": None, "activation": "linear"},
            {"name": "res_tower", "filters_init": 96, "filters_end": 384, "divisible_by": 32,
             "kernel_size": 5, "num_convs": 2, "dropout": 0.05, "pool_size": 2, "repeat": 7},
            {"name": "transformer_tower", "key_size": 64, "heads": 4, "num_position_features": 32,
             "dropout": 0.2, "mha_l2_scale": 1e-8, "l2_scale": 1e-8, "kernel_initializer": "he_normal", "repeat": 8},
            {"name": "unet_conv", "kernel_size": 3, "upsample_conv": True},
            {"name": "unet_conv", "kernel_size": 3, "upsample_conv": True},
            {"name": "unet_conv", "kernel_size": 3, "upsample_conv": True},
            {"name": "Cropping1D", "cropping": 64},
        ],
        "head": {"name": "final", "units": 5215, "activation": "softplus"},
    }

    print(f"=== Fold {fold}: Loading PyTorch model ===")
    model = Shorkie.from_tf_checkpoint(config, weights_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("  Using GPU")

    # ── Load golden TF predictions ──
    pred_path = f"/output/fold_{fold}/predictions.h5"
    print(f"Loading golden predictions from {pred_path}")
    with h5py.File(pred_path, "r") as f:
        n_seqs = f["sequences"].shape[0]
        print(f"  {n_seqs} sequences, predictions shape: {f['predictions'].shape}")

        BATCH_SIZE = 8
        all_abs_diffs = []
        all_correlations = []

        for start in range(0, n_seqs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_seqs)
            sequences = np.array(f["sequences"][start:end])   # (B, T, 4)
            tf_preds = np.array(f["predictions"][start:end])   # (B, T_out, targets)

            x = torch.from_numpy(sequences).permute(0, 2, 1).float()
            if torch.cuda.is_available():
                x = x.cuda()

            with torch.no_grad():
                pt_preds = model(x).cpu().numpy()

            abs_diff = np.abs(pt_preds - tf_preds)
            all_abs_diffs.append(abs_diff)

            for i in range(end - start):
                r = np.corrcoef(pt_preds[i].flatten(), tf_preds[i].flatten())[0, 1]
                all_correlations.append(r)

            if (start // BATCH_SIZE + 1) % 25 == 0 or end == n_seqs:
                print(f"  Processed {end}/{n_seqs} sequences")

    all_abs_diffs = np.concatenate(all_abs_diffs, axis=0)
    all_correlations = np.array(all_correlations)

    results = {
        "fold": fold,
        "n_sequences": n_seqs,
        "max_abs_diff": float(all_abs_diffs.max()),
        "mean_abs_diff": float(all_abs_diffs.mean()),
        "min_correlation": float(all_correlations.min()),
        "mean_correlation": float(all_correlations.mean()),
        "all_close_atol1e4": bool(np.all(all_abs_diffs < 1e-4)),
    }

    print(f"\n=== Results ===")
    print(f"  Sequences:       {results['n_sequences']}")
    print(f"  Max  abs diff:   {results['max_abs_diff']:.6e}")
    print(f"  Mean abs diff:   {results['mean_abs_diff']:.6e}")
    print(f"  Min  Pearson R:  {results['min_correlation']:.8f}")
    print(f"  Mean Pearson R:  {results['mean_correlation']:.8f}")
    print(f"  All close (1e-4): {results['all_close_atol1e4']}")

    return results


@app.local_entrypoint()
def main(fold: int = 0):
    result = test_equivalence.remote(fold)
    passed = result["all_close_atol1e4"] and result["min_correlation"] > 0.9999
    status = "PASSED" if passed else "FAILED"
    print(f"\n{status} — Fold {result['fold']}: {result['n_sequences']} sequences, "
          f"max diff {result['max_abs_diff']:.2e}, min R {result['min_correlation']:.8f}")
