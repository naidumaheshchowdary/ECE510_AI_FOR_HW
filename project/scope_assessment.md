# Project Scope Assessment
## ECE 410/510 HW4AI | Spring 2026
## Project: Fused Softmax + Layer Normalization Accelerator
## Updated: CF07 post-synthesis

---

## Current Scope (Confirmed)

The project scope remains as defined in M1 and M2:

- **Algorithm:** Fused softmax + layer normalization in a single streaming pass
  using online Welford algorithm and numerically stable softmax
- **Hardware:** 8-stage INT8 pipeline, AXI4-Lite control, AXI4-Stream 64-bit data
- **Target:** SKY130 HD PDK via OpenLane 2
- **Dimensions:** Fixed d=64, T=64 (professor's transformer_lm config)

---

## Synthesis Result Summary (CF07)

Yosys 0.9 synthesized `synth_top` to **92 generic cells** in 0.3 seconds:
- 27 `$adff` (pipeline registers) — as expected for 8-stage design
- 8 `$div` (S5 normalize) — **identified as critical path bottleneck**
- 8 `$mul` + 26 `$mux` — arithmetic and control logic

The 8 integer dividers in Stage 5 (softmax normalize) are the primary risk
for timing closure at 200 MHz on SKY130. This was not anticipated in M1/M2.

---

## Scope Adjustments for M3

**Adjusted:** Replace `$div` cells in S5 with reciprocal multiply-shift
(`recip = 65536 / running_sum` pre-computed once per row; output byte =
`(eb * recip) >> 8`). This is a local arithmetic change that does not
affect the interface, pipeline depth, or precision specification.

**Adjusted:** Clock target relaxed from 200 MHz to **100 MHz** for M3
synthesis attempt, with 200 MHz as a stretch goal once dividers are removed.

**Unchanged:** Pipeline depth (8 stages), INT8 precision, AXI4 interfaces,
d=64/T=64 dimensions, SKY130 PDK target.

---

## M3 Confidence Assessment

The design is synthesizable — Yosys completed with zero errors. The
critical path issue (division) has a known fix (reciprocal multiply).
M3 synthesis attempt on OpenLane 2 with proper SKY130 liberty is on
track for the May 24 deadline.
