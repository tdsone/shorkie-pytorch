"""Modal script for generating Shorkie TF inference test data.

Generates 1000 DNA sequences (500 random + 500 from sacCer3 yeast genome),
runs inference through each of 8 Shorkie fold checkpoints, and saves
predictions to a Modal Volume as HDF5 files.

Usage:
    # Run a single fold
    modal run modal_shorkie.py --fold 0

    # Run all 8 folds in parallel
    modal run modal_shorkie.py --all-folds
"""

import modal

app = modal.App("shorkie-inference")

# Container image with TF, baskerville, and sacCer3 genome baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libhdf5-dev", "samtools", "wget", "git"])
    .pip_install(
        [
            "baskerville @ git+https://github.com/calico/baskerville-yeast.git",
        ]
    )
    .run_commands(
        "mkdir -p /data",
        "wget -q https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz -O /data/sacCer3.fa.gz",
        "gunzip /data/sacCer3.fa.gz",
        "samtools faidx /data/sacCer3.fa",
    )
)

# Public GCS bucket with Shorkie checkpoints
gcs_mount = modal.CloudBucketMount(
    bucket_name="seqnn-share",
    bucket_endpoint_url="https://storage.googleapis.com",
    key_prefix="shorkie/",
    read_only=True,
)

# Persistent volume for outputs
output_volume = modal.Volume.from_name("shorkie-predictions", create_if_missing=True)

GENOME_PATH = "/data/sacCer3.fa"
SEQ_LEN = 16384
N_RANDOM = 500
N_GENOMIC = 500
BATCH_SIZE = 8
SEED = 42

# Nuclear chromosomes only (skip chrM)
NUCLEAR_CHROMS = [f"chr{x}" for x in [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
    "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
]]

NUM_SPECIES = 165
SPECIES_OFFSET = 5
R64_SPECIES_INDEX = 109


def generate_sequences(seq_len: int = SEQ_LEN, seed: int = SEED):
    """Generate 500 random + 500 genomic DNA sequences deterministically."""
    import numpy as np
    import pysam

    rng = np.random.RandomState(seed)
    dna_strings = []
    sources = []

    # --- 500 random sequences ---
    bases = np.array(list("ACGT"))
    for _ in range(N_RANDOM):
        seq = "".join(rng.choice(bases, size=seq_len))
        dna_strings.append(seq)
        sources.append("random")

    # --- 500 genomic sequences from sacCer3 ---
    fasta = pysam.FastaFile(GENOME_PATH)
    chrom_lengths = {
        name: length
        for name, length in zip(fasta.references, fasta.lengths)
        if name in NUCLEAR_CHROMS
    }

    # Build sampling weights proportional to available length
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
        sources.append("genomic")
        genomic_count += 1

    fasta.close()
    return dna_strings, sources


def one_hot_encode(dna_strings):
    """One-hot encode DNA sequences: A=0, C=1, G=2, T=3."""
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
    volumes={
        "/model": gcs_mount,
        "/output": output_volume,
    },
    timeout=3600,
)
def predict_fold(fold: int):
    """Run inference for a single Shorkie fold checkpoint."""
    import json
    import os

    import h5py
    import numpy as np

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    from baskerville.seqnn import SeqNN

    print(f"=== Fold {fold} ===")

    # Generate sequences (deterministic, same for every fold)
    print("Generating sequences...")
    dna_strings, sources = generate_sequences()
    sequences = one_hot_encode(dna_strings)
    print(f"  {len(dna_strings)} sequences, shape: {sequences.shape}")

    # Load model
    print("Loading model...")
    with open("/model/params.json") as f:
        params = json.load(f)

    # Build model with 170 input features to match pretrained species-encoded weights
    params["model"]["num_features"] = SPECIES_OFFSET + NUM_SPECIES  # 170
    seqnn = SeqNN(params["model"])
    weights_path = f"/model/f{fold}/model_best.h5"
    print(f"  Loading weights from {weights_path}")
    seqnn.restore(weights_path)

    # Species-encode input for inference
    sequences_170 = species_encode(sequences)
    print(f"  Species-encoded shape: {sequences_170.shape}")

    # Run batched inference
    print("Running inference...")
    n_seqs = len(dna_strings)
    n_batches = (n_seqs + BATCH_SIZE - 1) // BATCH_SIZE
    predictions = []

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_seqs)
        batch = sequences_170[start:end]
        pred = seqnn.model(batch, training=False).numpy()
        predictions.append(pred)
        if (i + 1) % 25 == 0 or i == n_batches - 1:
            print(f"  Batch {i + 1}/{n_batches}")

    predictions = np.concatenate(predictions, axis=0)
    print(f"  Predictions shape: {predictions.shape}")

    # Save to volume
    out_dir = f"/output/fold_{fold}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/predictions.h5"
    print(f"Saving to {out_path}...")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("sequences", data=sequences, compression="gzip")
        f.create_dataset("predictions", data=predictions, compression="gzip")
        dt = h5py.string_dtype()
        f.create_dataset("dna_strings", data=dna_strings, dtype=dt)
        f.create_dataset("source", data=sources, dtype=dt)
        f.attrs["fold"] = fold
        f.attrs["seed"] = SEED
        f.attrs["seq_len"] = SEQ_LEN
        f.attrs["n_random"] = N_RANDOM
        f.attrs["n_genomic"] = N_GENOMIC

    output_volume.commit()
    print(f"Fold {fold} complete: {out_path}")
    return {"fold": fold, "predictions_shape": list(predictions.shape)}


@app.local_entrypoint()
def main(fold: int = -1, all_folds: bool = False):
    if all_folds:
        print("Running all 8 folds in parallel...")
        results = list(predict_fold.map(range(8)))
        for r in results:
            print(f"  Fold {r['fold']}: predictions {r['predictions_shape']}")
    elif fold >= 0:
        result = predict_fold.remote(fold)
        print(f"Fold {result['fold']}: predictions {result['predictions_shape']}")
    else:
        print("Usage: modal run modal_shorkie.py --fold 0")
        print("       modal run modal_shorkie.py --all-folds")
