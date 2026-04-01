"""Extract model weights and generate consistent predictions for all 1000 sequences.

Builds the TF model with the full 170-channel species-encoded input (matching
the pretrained language model), loads the checkpoint, then generates predictions
for all 1000 sequences with proper species encoding for S. cerevisiae (R64).

Saves predictions as chunks of 100 for easy download.

Usage:
    modal run modal_extract_weights.py --fold 0
"""

import modal

app = modal.App("shorkie-extract-weights")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libhdf5-dev", "samtools", "wget", "git"])
    .pip_install([
        "baskerville @ git+https://github.com/calico/baskerville-yeast.git",
    ])
    .run_commands(
        "mkdir -p /data",
        "wget -q https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz -O /data/sacCer3.fa.gz",
        "gunzip /data/sacCer3.fa.gz",
        "samtools faidx /data/sacCer3.fa",
    )
)

gcs_mount = modal.CloudBucketMount(
    bucket_name="seqnn-share",
    bucket_endpoint_url="https://storage.googleapis.com",
    key_prefix="shorkie/",
    read_only=True,
)

output_volume = modal.Volume.from_name("shorkie-predictions", create_if_missing=True)

GENOME_PATH = "/data/sacCer3.fa"
SEQ_LEN = 16384
N_RANDOM = 500
N_GENOMIC = 500
BATCH_SIZE = 8
SEED = 42
CHUNK_SIZE = 100

NUCLEAR_CHROMS = [f"chr{x}" for x in [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
    "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
]]


def generate_sequences(seq_len=SEQ_LEN, seed=SEED):
    """Generate 500 random + 500 genomic DNA sequences (deterministic)."""
    import numpy as np
    import pysam

    rng = np.random.RandomState(seed)
    dna_strings = []

    bases = np.array(list("ACGT"))
    for _ in range(N_RANDOM):
        dna_strings.append("".join(rng.choice(bases, size=seq_len)))

    fasta = pysam.FastaFile(GENOME_PATH)
    chrom_lengths = {
        name: length
        for name, length in zip(fasta.references, fasta.lengths)
        if name in NUCLEAR_CHROMS
    }
    chroms = list(chrom_lengths.keys())
    available = np.array([chrom_lengths[c] - seq_len for c in chroms], dtype=float)
    available = np.maximum(available, 0)
    weights = available / available.sum()

    genomic_count = 0
    while genomic_count < N_GENOMIC:
        chrom = rng.choice(chroms, p=weights)
        max_start = chrom_lengths[chrom] - seq_len
        if max_start <= 0:
            continue
        start = rng.randint(0, max_start)
        seq = fasta.fetch(chrom, start, start + seq_len).upper()
        if "N" in seq:
            continue
        dna_strings.append(seq)
        genomic_count += 1

    fasta.close()
    return dna_strings


NUM_SPECIES = 165
SPECIES_OFFSET = 5
R64_SPECIES_INDEX = 109


def one_hot_encode(dna_strings):
    """Encode DNA strings as 4-channel one-hot (for saving to disk)."""
    import numpy as np
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    n = len(dna_strings)
    seq_len = len(dna_strings[0])
    arr = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i, seq in enumerate(dna_strings):
        for j, base in enumerate(seq):
            if base in mapping:
                arr[i, j, mapping[base]] = 1.0
    return arr


def species_encode(sequences_4ch):
    """Expand (N, T, 4) one-hot DNA to (N, T, 170) with S. cerevisiae species encoding."""
    import numpy as np
    n, seq_len, _ = sequences_4ch.shape
    total_channels = SPECIES_OFFSET + NUM_SPECIES  # 170
    arr = np.zeros((n, seq_len, total_channels), dtype=sequences_4ch.dtype)
    arr[:, :, :4] = sequences_4ch
    arr[:, :, SPECIES_OFFSET + R64_SPECIES_INDEX] = 1.0
    return arr


@app.function(
    image=image,
    gpu="L4",
    volumes={"/model": gcs_mount, "/output": output_volume},
    timeout=7200,
    memory=16384,
)
def extract_and_predict(fold: int):
    """Build TF model, extract weights, predict all 1000 sequences, save as chunks."""
    import json
    import os

    import h5py
    import numpy as np

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    from baskerville.seqnn import SeqNN

    with open("/model/params.json") as f:
        params = json.load(f)

    print(f"=== Fold {fold} ===")
    # Build model with 170 input features to match pretrained species-encoded weights
    params["model"]["num_features"] = SPECIES_OFFSET + NUM_SPECIES  # 170
    seqnn = SeqNN(params["model"])
    seqnn.restore(f"/model/f{fold}/model_best.h5")

    # Extract weights
    print("Extracting weights...")
    weights = {}
    for layer in seqnn.model.layers:
        for w in layer.weights:
            weights[w.name] = w.numpy()

    out_dir = f"/output/fold_{fold}"
    os.makedirs(out_dir, exist_ok=True)

    weights_path = f"{out_dir}/model_weights.h5"
    print(f"Saving {len(weights)} weight tensors to {weights_path}")
    with h5py.File(weights_path, "w") as f:
        for name, arr in weights.items():
            f.create_dataset(name, data=arr)

    # Generate all 1000 sequences (deterministic)
    print("Generating 1000 sequences...")
    dna_strings = generate_sequences()
    sequences = one_hot_encode(dna_strings)  # (N, T, 4) — saved to disk
    sequences_170 = species_encode(sequences)  # (N, T, 170) — fed to model
    print(f"  4-channel shape: {sequences.shape}")
    print(f"  170-channel shape: {sequences_170.shape}")

    # Run batched inference with species-encoded input
    print("Running inference...")
    n_seqs = len(sequences)
    predictions = []
    for i in range(0, n_seqs, BATCH_SIZE):
        batch = sequences_170[i : i + BATCH_SIZE]
        pred = seqnn.model(batch, training=False).numpy()
        predictions.append(pred)
        if (i // BATCH_SIZE + 1) % 25 == 0 or i + BATCH_SIZE >= n_seqs:
            print(f"  Batch {i // BATCH_SIZE + 1}/{(n_seqs + BATCH_SIZE - 1) // BATCH_SIZE}")

    predictions = np.concatenate(predictions, axis=0)
    print(f"  Predictions shape: {predictions.shape}")

    # Save as chunks
    chunks_dir = f"{out_dir}/consistent_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    n_chunks = (n_seqs + CHUNK_SIZE - 1) // CHUNK_SIZE

    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_seqs)
        chunk_path = f"{chunks_dir}/chunk_{c:03d}.h5"
        print(f"  Saving chunk {c}: seqs {start}-{end - 1}")
        with h5py.File(chunk_path, "w") as f:
            f.create_dataset("sequences", data=sequences[start:end], compression="gzip", compression_opts=1)
            f.create_dataset("predictions", data=predictions[start:end], compression="gzip", compression_opts=1)
            f.attrs["fold"] = fold
            f.attrs["chunk"] = c
            f.attrs["start_idx"] = start
            f.attrs["end_idx"] = end

    output_volume.commit()
    print("Done!")
    return {"predictions_shape": list(predictions.shape), "n_weights": len(weights), "n_chunks": n_chunks}


@app.local_entrypoint()
def main(fold: int = 0):
    result = extract_and_predict.remote(fold)
    print(f"Predictions: {result['predictions_shape']}, Weights: {result['n_weights']}, Chunks: {result['n_chunks']}")
