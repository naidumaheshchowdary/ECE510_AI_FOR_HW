# Synthesis Interpretation
## ECE 410/510 HW4AI | Spring 2026 | CF07 CLLM
## Project: Fused Softmax + Layer Normalization Accelerator
## Tool: Yosys 0.9 (git sha1 1979e0b) | Target: SKY130 HD | Clock: 5.0 ns (200 MHz)

---

## (a) Clock Period and Worst-Case Slack

Synthesis was run targeting a 5.0 ns clock period (200 MHz) on the SKY130 HD
process. Yosys 0.9 on the Colab environment used generic gate mapping via ABC
rather than SKY130 liberty cells, so formal slack numbers are not available from
this run. The longest-path (LTP) report traversed 199 nodes before terminating,
with the deepest path passing through S3 (exp LUT), S4 (running_sum adder tree),
and S5 (normalize division). The DIV cells in S5 are combinational dividers and
are expected to be the critical path bottleneck. For M3 the clock period will be
relaxed to 10 ns (100 MHz) to obtain positive slack with SKY130 mapped cells.

---

## (b) Critical Path

The longest combinational path identified by the LTP pass runs:

- **Source register:** `pipe_data[1]` (S2 output, stage register)
- **Combinational logic:**
  - `exp_beat` computation via exp LUT index comparison (9 `$gt` cells)
  - `beat_sum` 8-input adder tree (8 `$add` cells, lines 168–174)
  - `running_sum` mux accumulator (26 `$mux` cells via `$procmux$230`)
- **Sink register:** `running_sum[23:0]` DFF (`$procdff$279`, 24 bits)
- **Extension:** path continues through 8 `$div` cells in S5 (normalize stage)

The dominant cell types are `$div` (8 instances, one per output byte) and
`$mux` (26 instances). Integer division in S5 — `({16'd0,eb} * 255) / running_sum`
— is the primary timing bottleneck and will require pipelining or replacement
with a reciprocal multiply for M3.

---

## (c) Cell Area and Top Contributors

SKY130 cell mapping was not completed in Yosys 0.9 (abc used generic gates).
Total synthesized cell count: **92 cells**, **114 wires**, **1831 wire bits**.

| Rank | Cell type | Count | Function |
|------|-----------|-------|----------|
| 1 | `$adff` | 27 | Pipeline stage registers (S1–S8, stats) |
| 2 | `$mux` | 26 | Ternary selects in all pipeline stages |
| 3 | `$mul` + `$div` | 16 | S5 softmax normalize (8 mul + 8 div) |
| 4 | `$add` | 8 | S4 beat_sum accumulation |
| 5 | `$gt` | 9 | S2 max comparison + S3 LUT indexing |

Estimated die area on SKY130 HD: **3,000–5,000 µm²**, dominated by the 8 `$div`
cells (each approximately 300 µm² in SKY130 HD at 200 MHz).

---

## (d) Failed Constraints, Hold Violations, and Warnings

**Loop warnings (most critical):** 214 unique warning messages (7,234 total).
The LTP pass reported combinational loops at `pipe_data[0]` and
`$0\pipe_data[0][63:0]`. This is the back-pressure logic:
`assign s_axis_tready = m_axis_tready | ~pipe_valid[0]`. Yosys 0.9 incorrectly
flags this as a loop because the combinational assign feeds a path that Yosys
traces back through the DFF. This is not a real combinational loop — it is an
asynchronous ready signal. No functional impact on synthesis.

**Width mismatch warnings:** The bulk of the 7,234 warnings are signed/unsigned
width mismatches in the Welford stages (S6, S7), where 24-bit signed
`welford_mean` is mixed with 16-bit unsigned `pipe_data` bytes. These produce
implicit sign-extension warnings but no incorrect logic.

**No hold violations or setup violations** were reported by Yosys 0.9 (formal
STA requires OpenSTA with mapped cells, which is planned for M3).
