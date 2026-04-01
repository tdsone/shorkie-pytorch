# PyTorch Shorkie

A faithful PyTorch reimplementation of the Shorkie model (a yeast genomic deep learning model based on Baskerville).

## Species encoding

The original Shorkie model was pretrained as a multi-species language model on 165 Saccharomycetales species. The input to the first convolutional layer (`conv_dna`) has 170 channels:

- Channels 0-3: one-hot encoded DNA nucleotides (A, C, G, T)
- Channel 4: reserved
- Channels 5-169: one-hot species indicator (165 species)

For S. cerevisiae (R64), the species index is 109, so channel 114 is set to 1 across all positions.

The PyTorch model accepts standard 4-channel one-hot DNA input and expands it to the full 170-channel representation internally.

## Usage

```python
from shorkie import Shorkie
import json, torch

with open("data/shorkie_params.json") as f:
    config = json.load(f)

model = Shorkie.from_tf_checkpoint(config["model"], "data/checkpoint_f0.h5")
model.eval()

# Input: (batch, 4, seq_len) one-hot DNA
x = torch.randn(1, 4, 16384)
with torch.no_grad():
    y = model(x)  # (1, 896, 5215)
```

## Generating golden predictions

```bash
modal run modal_extract_weights.py --fold 0
```

## Running tests

```bash
pytest tests/
```
