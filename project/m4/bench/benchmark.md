# M4 Benchmark Results
## ECE 410/510 HW4AI | Spring 2026 | Milestone 4
## Fused Softmax + LayerNorm Accelerator

---

## Software Baseline (M1 — re-measured for M4)

**Platform:** DESKTOP-KDU44VU, Intel i5-5300U @ 2.30 GHz, 4 GB RAM
**Code:** professor's transformer_lm, pure NumPy FP64
**Method:** time.perf_counter(), 10 runs, warmup excluded

| Metric | Value |
|--------|-------|
| Execution time | 60.05 ms per forward pass |
| Throughput | 133.2 samples/sec |
| Peak memory | 13.8 MB |
| Precision | FP64 |

---

## Hardware Accelerator (M4 — projected from synthesis)

**Method: PROJECTED** — throughput from cycle count × synthesis clock.
No FPGA or cocotb measurement available. Per M4 spec Task 7 fallback.

**Projection basis:**
- Clock: 100 MHz (synthesis target, hash 9c2bb123ba)
- Pipeline: 8 stages, 1 row output per 8 cycles
- Row rate: 100 MHz ÷ 8 = 12,500,000 rows/sec
- Sample rate: 12,500,000 ÷ 64 rows/sample = **195,312 samples/sec**

| Metric | Value | Label |
|--------|-------|-------|
| Clock | 100 MHz | from synthesis |
| Cycles/row | 8 | from RTL |
| Throughput | 195,312 samples/sec | PROJECTED |
| Time/sample | 5.12 µs | PROJECTED |
| Synthesis cells | 132 | measured |
| LTP depth | 275 nodes | measured |

---

## Speedup vs M1 Baseline

**Speedup = 195,312 / 133.2 = 1,466× (PROJECTED)**

| Metric | SW Baseline | HW Accelerator | Ratio |
|--------|-------------|----------------|-------|
| Throughput | 133.2 samples/sec | 195,312 (PROJECTED) | 1,466× |
| Time/sample | 7.51 ms | 5.12 µs (PROJECTED) | 1,466× |

Reference vs original M1 (17.25 ms): 195,312 / 463.9 = **421× (PROJECTED)**

---

## Energy Comparison

| Metric | SW Baseline | HW Accelerator |
|--------|-------------|----------------|
| Power | ~15 W | ~58 µW (PROJECTED) |
| Time/sample | 60.05 ms | 5.12 µs (PROJECTED) |
| Energy/sample | 900.75 mJ | 0.30 nJ (PROJECTED) |
| Improvement | — | ~3×10⁹× (PROJECTED) |

All projected numbers are derived from synthesis cycle counts and manual
power estimates. See `synth/power_report.txt` for methodology.
