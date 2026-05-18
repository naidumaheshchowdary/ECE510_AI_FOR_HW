# M3 Synthesis Plan
## ECE 410/510 HW4AI | Spring 2026 | CF07 CLLM
## Project: Fused Softmax + Layer Normalization Accelerator

---

## Plan for Milestone 3 (due May 24)

**1. Replace S5 division with reciprocal multiply (highest priority)**
The 8 `$div` cells are the critical path bottleneck. Each
`({16'd0,eb} * 255) / running_sum` will be replaced with a pre-computed
reciprocal: `recip = 65536 / running_sum` (16-bit fixed-point), then
`out = (eb * recip) >> 8`. This eliminates all 8 `$div` cells and reduces
the critical path to a single multiply-shift chain.

**2. Relax clock target to 100 MHz (10 ns period)**
Yosys 0.9 produced 92 cells with 8 dividers. At 200 MHz the `$div` path
will not close timing on SKY130 HD. Dropping to 100 MHz gives 2× slack
budget and allows the dividers to be replaced iteratively in M3.

**3. Fix signed/unsigned width mismatches (214 unique warnings)**
The Welford stages S6/S7 mix `$signed(welford_mean)` with unsigned
`pipe_data` bytes. Each mismatch will be resolved with explicit casts
to eliminate the 7,234 warnings before OpenLane 2 full flow.

**4. Use OpenLane 2 with proper SKY130 liberty (not Yosys 0.9 generic)**
M3 will use OpenLane 2 Docker image to get real cell area, formal slack,
and power estimates that Yosys 0.9 could not produce.
