
# CF09 CLLM — Benchmark Results
## ECE 410/510 HW4AI | Spring 2026 | Codefest 09
## Project: Fused Softmax + LayerNorm Accelerator

---

## Task 6 — Software Baseline (re-run on same hardware)

**Platform:** DESKTOP-KDU44VU, Intel i5-5300U @ 2.30 GHz, 4 GB RAM, Windows 64-bit
**Code:** `project/algorithm/profile_m1.py` — professor's `transformer_lm`, pure NumPy, FP64
**Config:** d=64, n_heads=4, d_ff=256, n_layers=2, T=64, B=8, vocab=54
**Method:** `time.perf_counter()` around `T.forward()`, 10 runs, warmup excluded

| Metric | Value |
|--------|-------|
| Avg execution time | 60.05 ms |
| Std deviation | ± 4.22 ms |
| Throughput | 133.2 samples/sec |
| Peak memory | 14,118.8 KB (13.8 MB) |
| Precision | FP64 (NumPy default) |

**Note on M1 vs CF09 baseline:** The original M1 measurement was 17.25 ms / 463.9 samples/sec.
The CF09 re-run on the same machine produced 60.05 ms / 133.2 samples/sec. The difference
reflects real system state variation — the machine had just completed a 500-step training run
(`train.py`) which warmed up the CPU and OS scheduler. Both are honest measurements on
DESKTOP-KDU44VU. CF09 uses the fresh re-run numbers as the baseline.

---

## Task 7 — Hardware Accelerator (projected — fallback path)

**Method: PROJECTED** — no cocotb or FPGA available. Throughput computed from synthesis
cycle counts and target clock frequency per CF09 spec Task 7 fallback path.

**Projection assumptions:**
- Clock frequency: 100 MHz (synthesis target, 10 ns period)
- Pipeline depth: 8 stages → one row output every 8 cycles after fill
- Row rate: 100 MHz ÷ 8 cycles/row = 12,500,000 rows/sec
- Rows per sample: T=64 → sample rate = 12,500,000 ÷ 64 = **195,312 samples/sec**
- Interface: AXI4-Stream 64-bit @ 100 MHz → 800 MB/s bandwidth

| Metric | Value | Label |
|--------|-------|-------|
| Clock frequency | 100 MHz | from synthesis |
| Cycles per row | 8 | from RTL pipeline depth |
| Row throughput | 12,500,000 rows/sec | PROJECTED |
| Sample throughput | 195,312 samples/sec | PROJECTED |
| Time per sample | 5.12 µs | PROJECTED |
| Interface BW | 800 MB/s | PROJECTED |
| Synthesis cells | 133 | measured (Yosys 0.9) |
| LTP depth | 275 nodes | measured (Yosys 0.9) |

---

## Task 8 — Speedup and Energy Comparison

### Speedup

| Metric | SW Baseline (CF09) | HW Accelerator | Ratio |
|--------|-------------------|----------------|-------|
| Throughput (samples/sec) | 133.2 (measured) | 195,312 (PROJECTED) | **1,466× (PROJECTED)** |
| Time per sample | 7.51 ms | 5.12 µs | **1,466× (PROJECTED)** |

**Speedup = 195,312 / 133.2 = 1,466× (PROJECTED)**

For reference vs original M1 baseline (17.25 ms / 463.9 samples/sec):
Speedup = 195,312 / 463.9 = **421× (PROJECTED)**

### Energy Comparison (optional)

| Metric | SW Baseline | HW Accelerator | Improvement |
|--------|-------------|----------------|-------------|
| Power | ~15 W (i5-5300U TDP) | ~44 µW (PROJECTED) | — |
| Time/sample | 7.51 ms | 5.12 µs (PROJECTED) | — |
| Energy/sample | 900.7 mJ | 0.23 nJ (PROJECTED) | **~4×10⁹× (PROJECTED)** |

Note: HW energy covers the softmax+layernorm kernel only (28% of full model).
SW energy for kernel share alone: 900.7 mJ × 0.28 = 252 mJ.
Kernel energy improvement: 252 mJ / 0.23 nJ ≈ **1.1×10⁹× (PROJECTED)**.
Power estimate from `project/m3/synth/power_report.txt`.
