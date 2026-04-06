# PyTorch Shorkie

A PyTorch reimplementation of the Shorkie model (a yeast sequence-to-expression model by Chao et al.).
- [Paper](https://www.biorxiv.org/content/10.1101/2025.09.19.677475v1)
- [Github](https://github.com/calico/shorkie-paper)

## Usage

### Prerequisites
1. Clone the repo
2. Run `uv sync` to install all required packages.
3. Get a model checkpoint (see [Downloading checkpoints](#downloading-checkpoints))

### Make your first prediction
See [demo.ipynb](demo.ipynb) for a runnable example.
```python
from shorkie import Shorkie
import json
import random
import torch

with open("data/shorkie_params.json") as f:
    config = json.load(f)

model = Shorkie.from_tf_checkpoint(config["model"], "data/checkpoint_f0.h5")
model.eval()

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def one_hot_encode(seq: str) -> torch.Tensor:
    t = torch.zeros(4, len(seq))
    for i, ch in enumerate(seq):
        t[BASE_TO_IDX[ch], i] = 1.0
    return t


def random_dna_sequence(length: int) -> str:
    return "".join(random.choices(BASES, k=length))


sequences = torch.stack([one_hot_encode(random_dna_sequence(16384)) for _ in range(5)])

with torch.no_grad():
    y = model(sequences)  # (5, 896, 5215)

print(f"{y.shape=}")
```

## Downloading checkpoints

The original TF/Keras checkpoints are hosted on Google Cloud Storage by the [Shorkie paper authors](https://github.com/calico/shorkie-paper). Download the fine-tuned Shorkie checkpoints (one per cross-validation fold):

- [f0](https://storage.googleapis.com/seqnn-share/shorkie/f0/model_best.h5) | [f1](https://storage.googleapis.com/seqnn-share/shorkie/f1/model_best.h5) | [f2](https://storage.googleapis.com/seqnn-share/shorkie/f2/model_best.h5) | [f3](https://storage.googleapis.com/seqnn-share/shorkie/f3/model_best.h5) | [f4](https://storage.googleapis.com/seqnn-share/shorkie/f4/model_best.h5) | [f5](https://storage.googleapis.com/seqnn-share/shorkie/f5/model_best.h5) | [f6](https://storage.googleapis.com/seqnn-share/shorkie/f6/model_best.h5) | [f7](https://storage.googleapis.com/seqnn-share/shorkie/f7/model_best.h5)

The pretrained language model (Shorkie LM) is also available but we do not implement the language model here (just the full seq2expr model):

- [Shorkie LM](https://storage.googleapis.com/seqnn-share/shorkie_lm/train/model_best.h5)

Place the downloaded checkpoint in the `data/` directory, e.g.:

```bash
wget -O data/checkpoint_f0.h5 https://storage.googleapis.com/seqnn-share/shorkie/f0/model_best.h5
```

## Species encoding

The original Shorkie model was pretrained as a multi-species language model on 165 Saccharomycetales species. The input to the first convolutional layer (`conv_dna`) has 170 channels:

- Channels 0-3: one-hot encoded DNA nucleotides (A, C, G, T)
- Channel 4: reserved
- Channels 5-169: one-hot species indicator (165 species)

For S. cerevisiae (R64), the species index is 109, so channel 114 is set to 1 across all positions.

The PyTorch model accepts standard 4-channel one-hot DNA input and expands it to the full 170-channel representation internally.


## Generating predictions with the Tensorflow-based model

```bash
modal run modal_extract_weights.py --fold 0
```

## Running tests

```bash
pytest tests/
```
