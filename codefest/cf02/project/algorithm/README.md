# Transformer language model (pure NumPy)

A character-level transformer language model implemented entirely in NumPy, with no deep learning framework dependencies. Every operation -- multi-head self-attention, layer normalization, GELU activations, causal masking, and backpropagation -- is written from first principles.

## What is implemented

- Multi-head causal self-attention with full backward pass
- Pre-norm transformer blocks (layer norm before attention and feed-forward)
- GELU activation with analytic gradient
- Adam optimizer with weight decay
- Character-level tokenizer
- Temperature sampling with top-k filtering
- Checkpoint save/load

## Files

```
transformer_lm/
  transformer.py   -- model, layers, optimizer, training loop, generation
  train.py         -- CLI entry point
  data/
    sample.txt     -- excerpt from Alice in Wonderland (public domain)
```

## Dependencies

```
numpy
matplotlib   # optional, for loss plots
```

Install with:

```bash
pip install numpy matplotlib
```

## Quickstart

Train a small model on the included sample text for 500 steps:

```bash
python train.py --config small --steps 500 --generate --prompt "Alice"
```

Expected output after 500 steps (small config, character-level):

```
Step    1 | loss 5.34
Step   50 | loss 2.49
...
Step  500 | loss 2.21

--- Generated text ---
Alice the ther anthe she had the ...
```

Loss around 2.0-2.3 on 4 KB of text with a small model is expected. Provide more data and a medium config to see meaningful generation.

## Scaling the problem

The problem can be scaled along several axes:

### Dataset size

Replace `data/sample.txt` with any plain text file. Larger corpora expose the model to a wider vocabulary and longer-range dependencies.

```bash
# Download a larger corpus (e.g. from Project Gutenberg)
curl -o data/shakespeare.txt https://www.gutenberg.org/cache/epub/100/pg100.txt
python train.py --text data/shakespeare.txt --config medium --steps 5000 --generate
```

### Model size

Three named configurations are available:

| Config | d_model | Heads | Layers | d_ff | Seq len |
|--------|---------|-------|--------|------|---------|
| small  | 64      | 4     | 2      | 256  | 64      |
| medium | 128     | 4     | 4      | 512  | 128     |
| large  | 256     | 8     | 6      | 1024 | 256     |

```bash
python train.py --config large --steps 2000 --text data/shakespeare.txt
```

### Sequence length

Longer sequences make attention O(T^2) in time and memory, which quickly becomes the dominant cost. The `seq_len` parameter in any config dict controls this.

### Depth and width

Edit `CONFIGS` in `train.py` or pass a custom config dict in Python:

```python
from transformer import train, generate

config = {
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_layers": 8,
    "seq_len": 512,
    "batch_size": 4,
    "lr": 3e-4,
}

with open("data/sample.txt") as f:
    text = f.read()

params, tokenizer, losses = train(text, config, n_steps=2000, checkpoint_path="my_model.pkl")
```

## Python API usage

### Training

```python
from transformer import train, generate

with open("data/sample.txt") as f:
    text = f.read()

config = {
    "d_model": 64, "n_heads": 4, "d_ff": 256, "n_layers": 2,
    "seq_len": 64, "batch_size": 8, "lr": 3e-3,
}

params, tokenizer, losses = train(
    text, config,
    n_steps=500,
    log_every=50,
    checkpoint_path="checkpoint.pkl"
)
```

### Generation from a checkpoint

```python
from transformer import load_and_generate

text = load_and_generate(
    "checkpoint.pkl",
    prompt="The rabbit",
    max_new=200,
    temperature=0.8,
    top_k=20,
)
print(text)
```

### Generation from in-memory params

```python
from transformer import generate

output = generate(
    "Once upon a time",
    params, tokenizer, config,
    max_new=300,
    temperature=1.0,   # higher = more random
    top_k=None,        # None = full distribution
)
```

## Computational cost

| Config | Parameters (approx.) | Memory (float64) | Steps/sec (CPU) |
|--------|-----------------------|-------------------|-----------------|
| small  | ~200K                 | ~6 MB             | ~15-30          |
| medium | ~1.5M                 | ~46 MB            | ~3-8            |
| large  | ~10M                 | ~310 MB           | <1              |

The dominant cost is the attention matrix (N_heads x T x T), which grows quadratically with sequence length. Increasing `seq_len` from 64 to 512 multiplies attention cost by 64x.

## Notes on correctness

The implementation uses pre-layer-norm (GPT-2 style), which trains more stably than post-norm at small scales. The causal mask prevents each token from attending to future positions. Backpropagation is derived analytically for every operation; there is no numerical differentiation.

To verify gradients numerically:

```python
import numpy as np
from transformer import forward, cross_entropy_loss, backward, init_params, CharTokenizer, get_batch

# Finite-difference gradient check on a small model
config = {"d_model": 16, "n_heads": 2, "d_ff": 32, "n_layers": 1,
          "seq_len": 8, "batch_size": 2, "vocab_size": 10}
params = init_params(10, 8, 16, 2, 32, 1, seed=0)
batch = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2, dtype=np.int32)
logits, caches = forward(batch, params, config)
loss, dlogits = cross_entropy_loss(logits, batch)
grads = backward(dlogits, caches, params, config)

# Check one parameter numerically
eps = 1e-4
key = "ln_f_g"
for i in range(min(4, params[key].size)):
    params[key].flat[i] += eps
    l1, _ = cross_entropy_loss(*forward(batch, params, config)[:1], batch)
    # (unpack correctly)
    logits2, _ = forward(batch, params, config)
    l1, _ = cross_entropy_loss(logits2, batch)
    params[key].flat[i] -= 2 * eps
    logits3, _ = forward(batch, params, config)
    l2, _ = cross_entropy_loss(logits3, batch)
    params[key].flat[i] += eps
    numerical = (l1 - l2) / (2 * eps)
    analytical = grads[key].flat[i]
    print(f"[{key}][{i}] numerical={numerical:.6f}  analytical={analytical:.6f}")
```
