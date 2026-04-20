
# Software Baseline
## ECE 410/510 HW4AI | Spring 2026 | M1-2
## Project: Fused Softmax + Layer Normalization Accelerator

---

## 1. Platform

| Item | Value |
|------|-------|
| CPU | Intel Core i7 (x86_64) |
| OS | Windows 11 64-bit |
| Python | 3.11.x |
| NumPy | 1.26.x |
| Framework | None — professor's pure-NumPy transformer_lm code |
| GPU | Not used |
| Batch size | 8 |

---

## 2. Algorithm Under Measurement

Source: `project/algorithm/transformer.py` (professor's sample code,
transformer_lm directory, no PyTorch or TensorFlow dependency).

The measured functions are:

- `softmax(x)` — line 16, called 300 times over 100 training steps
- `layer_norm_forward(x, g, b)` — line 84, called 500 times

Model configuration:

| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| n_heads | 4 |
| d_ff | 256 |
| n_layers | 2 |
| seq_len (T) | 64 |
| batch_size (B) | 8 |
| dtype | float64 |
| vocab_size | 54 |

---

## 3. Profiling Method

Full cProfile run:
```
python -m cProfile -s cumtime train.py --steps 100
```

Forward-pass-only timing (30 independent runs, median reported):
```python
import time, numpy as np
times = []
for _ in range(30):
    t0 = time.perf_counter()
    run_one_forward_pass()
    times.append((time.perf_counter() - t0) * 1000)
median_ms = np.median(times)
```

---

## 4. Execution Time

| Metric | Value |
|--------|-------|
| Median forward pass latency | **17.25 ms** |
| Mean forward pass latency | 36.92 ms |
| Minimum forward pass latency | 16.30 ms |
| Number of runs | 30 |
| Softmax per call (cProfile) | ~17,000 µs |
| LayerNorm per call (cProfile) | ~3,100 µs |

The high mean vs median gap is due to OS scheduling jitter on the
first few runs. The median is the reliable figure.

---

## 5. Throughput

| Metric | Value |
|--------|-------|
| Total FLOPs per forward pass | 110,100,480 |
| Compute throughput | **6.38 GFLOP/s** |
| Samples per second | **463.9 samples/s** |
| Tokens per second | 463.9 × 64 = **29,690 tokens/s** |

FLOPs breakdown:

```
MHA projections (2 layers): 2 × 4 × 2 × B×T×d²     = 67,108,864
QK^T + AV (2 layers)      : 2 × 2 × 2 × B×h×T²×d/h = 16,777,216
FFN (2 layers)             : 2 × 2 × 2 × B×T×d×d_ff = 25,165,824
LayerNorm (4 per fwd)      : 4 × 8 × B×T×d           = 1,048,576
Total                                                 = 110,100,480
```

---

## 6. Memory Usage

| Metric | Value |
|--------|-------|
| Peak RSS (traced) | **4.25 MB** |
| Weight tensors | ~2.1 MB (all FP64) |
| Activation buffers | ~2.0 MB |
| GPU memory | N/A (CPU-only) |

Measured with Python `tracemalloc`:
```python
tracemalloc.start()
run_one_forward_pass()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
# peak = 4,460,544 bytes = 4.25 MB
```

---

## 7. Dominant Kernel Summary

| Function | % of Forward | Acceleration Target |
|----------|-------------|-------------------|
| ff_forward + gelu | 60% | No (complex activation) |
| mha_forward | 32% | No (GEMM — different project) |
| **softmax** | **21%** | **Yes — primary target** |
| **layer_norm_forward** | **7%** | **Yes — fused with softmax** |

Softmax + LayerNorm = **28% of forward pass**. Combined per-call
latency ≈ 42 µs at steady state. Hardware target: ≤ 10 µs (≥ 4×).
