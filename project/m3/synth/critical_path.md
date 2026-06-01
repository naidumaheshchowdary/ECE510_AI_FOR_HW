# Critical Path Analysis
## ECE 410/510 HW4AI | Spring 2026 | Milestone 3
## Design: top (interface + compute_core_m3)

---

## Critical Path Identification

**Start register:** `u_interface.u_core.final_sum[23:0]`
The 24-bit softmax denominator register (`$procdff$579`), clocked on
the rising edge of `clk`. This register captures the complete row sum
at the end of each 8-beat input row (when `pipe_last[3]` fires), and
holds it stable while S5 normalizes all 8 output beats.

**End register:** `u_interface.u_core.pipe_data[4][63:0]`
The Stage 5 output register (`$procdff$575`), capturing the normalized
softmax result after division.

**Logic stages between start and end register:**

1. `final_sum[23:0]` DFF output (24 bits, node 268 in LTP)
2. `$ne` cell (`$ne$/content/compute_core_m3.sv:138$341`) — 24-bit
   not-equal to zero, produces zero-gate enable (~0.3 ns)
3. 8× `$div` cells — one per output byte, each computing
   `({16'd0, eb_n} * 24'd255) / final_sum` (~8–12 ns, BOTTLENECK)
4. 8× `$ternary` — zero-gate: `final_sum!=0 ? div_result : 8'd0`
   (node 270: `norm_beat[0]` via `$ternary$138$345`)
5. `$ternary` mux — `pipe_valid[3] ? norm_beat : pipe_data[3]`
   (node 271: `$ternary$152$383`)
6. `pipe_data[4][63:0]` DFF setup time (~0.2 ns)

**Why this is the critical path:**
Integer division maps to an iterative subtraction circuit (~24 gate
levels for a 24-bit divisor) on SKY130 HD. Each `$div` cell is
estimated at 8–12 ns at TT/25C/1.8V. All 8 divisions share the same
`final_sum` divisor and run in parallel, so the path delay equals
one `$div` plus surrounding logic: approximately **9–13 ns total**.
This exceeds the 10 ns target clock period and fails at 200 MHz (5 ns).

**What changed vs previous run:**
The `final_sum` register was added to fix a Verilog non-blocking
assignment race where both `running_sum <= running_sum + beat_sum`
and `running_sum <= 24'd0` fired on the same clock edge — the reset
always won, making beat 7 produce zero output. The fix captures the
complete accumulated sum into `final_sum` before the reset fires.
This adds 50 nodes to the LTP (nodes 219–268: the `final_sum` DFF
chain) and increases total LTP from 225 to 275.

**What would shorten it:**
Replace `eb / final_sum` with a reciprocal multiply:
```systemverilog
wire [15:0] recip = 16'hFFFF / final_sum[15:0];  // computed once per row
norm_byte = (eb * recip) >> 8;                    // per byte
```
This eliminates all 8 `$div` cells and reduces the critical path
from ~9–13 ns to ~2.5 ns, enabling 200 MHz timing closure. This is
the primary M4 optimization task (see `project/remaining_tasks.md`).
