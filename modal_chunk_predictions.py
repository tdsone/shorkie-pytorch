"""Chunk the large predictions H5 into groups of 100 sequences.

Usage:
    modal run modal_chunk_predictions.py --fold 0
"""

import modal

app = modal.App("shorkie-chunk")

output_volume = modal.Volume.from_name("shorkie-predictions")

image = modal.Image.debian_slim(python_version="3.11").pip_install(["h5py", "numpy"])


@app.function(
    image=image,
    volumes={"/output": output_volume},
    memory=4096,
    timeout=600,
)
def chunk_predictions(fold: int, chunk_size: int = 100):
    """Split predictions.h5 into chunks of chunk_size sequences."""
    import os
    import h5py
    import numpy as np

    src = f"/output/fold_{fold}/predictions.h5"
    out_dir = f"/output/fold_{fold}/chunks"
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(src, "r") as f:
        n_seqs = f["sequences"].shape[0]
        print(f"Total sequences: {n_seqs}")
        print(f"Predictions shape: {f['predictions'].shape}")
        print(f"Sequences shape: {f['sequences'].shape}")

        n_chunks = (n_seqs + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_seqs)
            chunk_path = f"{out_dir}/chunk_{i:03d}.h5"

            print(f"  Writing chunk {i}: sequences {start}-{end-1} -> {chunk_path}")
            with h5py.File(chunk_path, "w") as out:
                out.create_dataset(
                    "sequences", data=f["sequences"][start:end], compression="gzip", compression_opts=9
                )
                out.create_dataset(
                    "predictions", data=f["predictions"][start:end], compression="gzip", compression_opts=9
                )
                out.attrs["fold"] = fold
                out.attrs["chunk"] = i
                out.attrs["start_idx"] = start
                out.attrs["end_idx"] = end

    output_volume.commit()
    print(f"Done — {n_chunks} chunks written")
    return n_chunks


@app.local_entrypoint()
def main(fold: int = 0):
    n = chunk_predictions.remote(fold)
    print(f"Created {n} chunks for fold {fold}")
