"""Download golden TF predictions and fine-tuned checkpoints from Modal/GCS.

Usage:
    # Download predictions for fold 0
    modal run modal_download_predictions.py --fold 0

    # Download predictions for all folds
    modal run modal_download_predictions.py --all-folds
"""

import modal

app = modal.App("shorkie-download")

# Persistent volume with TF predictions (created by modal_shorkie.py)
output_volume = modal.Volume.from_name("shorkie-predictions")

# GCS bucket with fine-tuned checkpoints
gcs_mount = modal.CloudBucketMount(
    bucket_name="seqnn-share",
    bucket_endpoint_url="https://storage.googleapis.com",
    key_prefix="shorkie/",
    read_only=True,
)

image = modal.Image.debian_slim(python_version="3.11").pip_install(["h5py", "numpy"])


@app.function(
    image=image,
    volumes={
        "/output": output_volume,
        "/model": gcs_mount,
    },
    timeout=600,
)
def download_fold(fold: int):
    """Read predictions HDF5 from volume and return as bytes, plus checkpoint."""
    import io
    import shutil

    results = {}

    # Read predictions
    pred_path = f"/output/fold_{fold}/predictions.h5"
    with open(pred_path, "rb") as f:
        results["predictions"] = f.read()

    # Read fine-tuned checkpoint
    ckpt_path = f"/model/f{fold}/model_best.h5"
    with open(ckpt_path, "rb") as f:
        results["checkpoint"] = f.read()

    # Read params
    with open("/model/params.json", "rb") as f:
        results["params"] = f.read()

    return results


@app.local_entrypoint()
def main(fold: int = -1, all_folds: bool = False):
    import os

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    folds = list(range(8)) if all_folds else [fold] if fold >= 0 else []

    if not folds:
        print("Usage: modal run modal_download_predictions.py --fold 0")
        print("       modal run modal_download_predictions.py --all-folds")
        return

    for f_idx in folds:
        print(f"Downloading fold {f_idx}...")
        result = download_fold.remote(f_idx)

        pred_path = os.path.join(out_dir, f"predictions_f{f_idx}.h5")
        with open(pred_path, "wb") as f:
            f.write(result["predictions"])
        print(f"  Saved predictions to {pred_path}")

        ckpt_path = os.path.join(out_dir, f"checkpoint_f{f_idx}.h5")
        with open(ckpt_path, "wb") as f:
            f.write(result["checkpoint"])
        print(f"  Saved checkpoint to {ckpt_path}")

        # Save params (only once)
        params_path = os.path.join(out_dir, "params_gcs.json")
        if not os.path.exists(params_path):
            with open(params_path, "wb") as f:
                f.write(result["params"])
            print(f"  Saved params to {params_path}")

    print("Done!")
