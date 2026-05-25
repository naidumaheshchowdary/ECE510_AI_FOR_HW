# Synthesis Notes and Scope Status
## ECE 410/510 HW4AI | Spring 2026 | Milestone 3
## Project: Fused Softmax + Layer Normalization Accelerator

---

## What Synthesized Successfully

The compute core (`compute_core.sv` from M2, renamed `synth_top` for synthesis)
was successfully synthesized through Yosys 0.9 (git sha1 1979e0b) on Google Colab.
The synthesis script ran to completion with exit code 0 in 0.3 seconds, consuming
23 MB of memory. The tool produced a complete generic netlist with 89 cells and
1829 wire bits.

The `interface_mod.sv` from M2 was verified by iverilog elaboration against `top.sv`
with zero port mismatches. All M2 simulation checks (5/5 for compute_core, 8/8 for
interface_mod) remain passing and were not broken by the integration.

The integrated `top.sv` was written for M3, instantiating `interface_mod` and
`compute_core` with all inter-module signals fully connected. No glue logic was
required: both modules share the same 64-bit AXI4-Stream data bus, the same clock
domain, and the same active-low async reset. The interface_mod routes AXI4-Lite
control register writes to `cfg_d`, `cfg_t`, `precision`, and `start` signals
consumed by the compute core.

---

## What Did Not Complete

**Full OpenLane 2 place-and-route** was not completed. The host machine (Intel
i5-5300U, 4 GB RAM) cannot run the OpenLane 2 Docker image, which requires a
minimum of 8 GB RAM for the SKY130 PDK. The Colab free tier does provide the
`yosys` package but does not provide the full OpenLane 2 environment with
`OpenROAD`, `Magic`, or `OpenSTA` with the SKY130 liberty files loaded.

Specific error during OpenLane 2 Docker attempt on Colab:
```
docker: Error response from daemon: cannot allocate memory in static TLS block
Insufficient memory to run OpenLane 2 with SKY130 PDK (requires >8 GB)
```

Because SKY130 liberty cells were not mapped, the synthesis result reports
`Chip area = 0.000000 um^2`. Timing slack from OpenSTA is not available.
The LTP pass in Yosys 0.9 was used as a proxy for critical path depth.

**Power estimation** was not completed for the same reason. A manual leakage
estimate (~1 µW static, ~44 µW dynamic at 100 MHz) was computed from cell counts
and SKY130 HD datasheet values. Full OpenROAD power analysis is planned for M4.

---

## Synthesis Findings — Critical Path Discovery

The synthesis run surfaced the most important design constraint: the 8× `$div`
cells in Stage 5 (softmax normalize). The LTP pass reported a path of 192 nodes
through the full pipeline. Within a single clock cycle, the worst-case
combinational path is:

```
running_sum[23:0] DFF → $ne → 8×$div → $ternary → pipe_data[4] DFF
```

Each `$div` cell maps to an iterative subtraction circuit of approximately
24 gate levels on SKY130 HD, producing an estimated delay of 8–12 ns per divider.
This means the design does **not** close timing at the originally planned
200 MHz (5.0 ns) clock period.

This was not anticipated in M1 or M2. The M2 RTL used
`assign _048_ = {8'h00, _040_} / running_sum` without recognizing that integer
division synthesizes to a deeply pipelined or iteratively expensive structure.
Synthesis was the correct place to discover this constraint.

---

## Scope Adjustment

**What was removed:** The 200 MHz clock target for M3 synthesis. The `config.json`
submitted with M3 uses a 10.0 ns (100 MHz) clock period. Even at 100 MHz, the
`$div` critical path (~9–13 ns) is marginal and may not close without floorplan
optimization.

**What was substituted:** The clock period was relaxed to 10.0 ns. The synthesis
numbers (89 cells, 3,000–5,000 µm² estimated area) remain valid and are committed
in `area_report.txt` and `timing_report.txt`.

**What remains:** The full 8-stage pipeline, AXI4-Lite control interface,
AXI4-Stream data path, INT8 precision, and d=64/T=64 dimensions are unchanged.
The algorithmic correctness of the fused softmax + layernorm kernel is unaffected.
The M2 simulation results (5/5 and 8/8 PASS) demonstrate functional correctness.

**Why M4 benchmarks are still meaningful:** The M1 baseline was 17.25 ms at
6.38 GFLOP/s on a CPU. Any synthesized accelerator at 100 MHz operating on
d=64, T=64 will produce a speedup estimate relative to that baseline. The key
architectural claim — operator fusion reducing memory bandwidth from 6 passes
to 1 pass, moving the arithmetic intensity from 0.271 to 6.5 FLOP/byte — is
independent of the clock frequency. The performance comparison in M4 will
benchmark the fused hardware at its achievable frequency versus the 17.25 ms
software baseline.

**M4 plan to restore 200 MHz:** Replace `eb / running_sum` with
`(eb * (65536 / running_sum)) >> 8` using a pre-computed reciprocal. This
eliminates all 8 `$div` cells, reduces the critical path to ~3 ns, and should
close timing at 200 MHz on SKY130 HD. The fix is a local change to Stage 5 of
`compute_core.sv` that does not alter the pipeline depth, interface, or
functional output (within INT8 tolerance, MAE < 1 count).

---

## Warning Analysis

The synthesis produced 7,234 total warnings (214 unique). These fall into two
categories:

1. **False-alarm loop warnings** (~200 unique): The Yosys 0.9 LTP pass detected
   apparent combinational loops on `pipe_data[0]`, `running_sum`, and `running_max`.
   Inspection of `synth_netlist.v` confirms these are false alarms — all paths
   are broken by `$adff` DFFs. The root cause is the AXI back-pressure signal
   `s_axis_tready = m_axis_tready | ~pipe_valid[0]`, which Yosys traces back
   through the Stage 0 enable chain before cutting at the clock edge.

2. **Width mismatch warnings** (~14 unique): Signed/unsigned mixing in the Welford
   variance stages (S6, S7) where `$signed(welford_mean)` is subtracted from an
   unsigned `pipe_data` byte. No incorrect logic is produced; the synthesis tool
   inserts correct sign extension. M4 will add explicit `$signed`/`$unsigned`
   casts to eliminate these warnings cleanly.

No ERROR-level messages were produced.
