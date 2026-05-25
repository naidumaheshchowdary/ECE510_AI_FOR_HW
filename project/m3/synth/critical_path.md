# Critical Path Analysis
## ECE 410/510 HW4AI | Spring 2026 | Milestone 3
## Design: top (interface + compute_core_m3)

---

## Critical Path Identification

**Start register:** `u_interface.u_core.running_sum[23:0]`
The 24-bit softmax denominator accumulator register (`$procdff$572`),
clocked on the rising edge of `clk` inside `compute_core_m3`.

**End register:** `u_interface.u_core.pipe_data[4][63:0]`
The Stage 5 output register (`$procdff$569`), capturing the normalized
softmax result after division.

**Logic stages between start and end (per-cycle combinational path):**

1. `running_sum[23:0]` DFF output (24 bits)
2. `$ne` cell (`$ne$/content/compute_core_m3.sv:118$339`) — 24-bit
   not-equal to zero, produces 1-bit zero-gate enable `_057_`
3. 8× `$div` cells — one per output byte:
   `norm_beat[7:0]  = ({16'd0,eb0} * 24'd255) / running_sum`
   through
   `norm_beat[63:56] = ({16'd0,eb7} * 24'd255) / running_sum`
   These 8 divisions are the dominant delay (~8-12 ns each on SKY130 HD)
4. 8× `$ternary` — zero-gate: `running_sum!=0 ? div_result : 8'd0`
   (node 220: `norm_beat[56]` via `$ternary$/content/compute_core_m3.sv:125$378`)
5. `$ternary` mux — `pipe_valid[3] ? norm_beat : pipe_data[3]`
   (node 221: `$ternary$/content/compute_core_m3.sv:131$381`)
6. `pipe_data[4][63:0]` DFF setup time

**Why this is the critical path:**
Integer division in hardware maps to an iterative subtraction circuit
(non-restoring divider) requiring approximately 24 gate levels for a
24-bit divisor. Each `$div` cell on SKY130 HD at TT/25C/1.8V is
estimated at 8–12 ns. Since all 8 divisions share the same `running_sum`
divisor and operate in parallel, the critical path delay equals one
`$div` plus surrounding logic — approximately 9–13 ns total.

At 10.0 ns (100 MHz) this is marginal. At 5.0 ns (200 MHz) it fails
by 4–8 ns. Synthesis confirmed with Yosys 0.9 LTP depth = 225 nodes.

**What would shorten it:**
Replace all 8 `$div` operations with a reciprocal multiply:
```systemverilog
// Compute reciprocal once per row (in S4, when pipe_last arrives)
reg [15:0] recip;
always @(posedge clk) begin
    if (pipe_last[3] && running_sum != 0)
        recip <= 16'hFFFF / running_sum[15:0];
end

// Use multiply-shift instead of division in S5
norm_beat[7:0] = (eb0 * recip) >> 8;
```
This eliminates all 8 `$div` cells (~300 µm² each, ~8-12 ns delay)
and replaces them with 8 `$mul` + shift (~50 µm², ~2.5 ns), reducing
the critical path to approximately 3 ns and enabling 200 MHz closure.
This is the primary M4 optimization.

---

## LTP Context

The Yosys LTP reported 225 total nodes through the full integrated design.
This is 33 nodes more than the compute_core-alone path (192 nodes in CF07),
because the flattened `top` includes the reset propagation path through
`pipe_valid[0]` → `tready` → `logic_and` at the pipeline entry.
The per-cycle critical path (nodes 148–222) is identical in both cases —
it is the `running_sum` DFF to `pipe_data[4]` DFF segment that limits timing.
