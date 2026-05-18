
# Synthesis Interpretation
ECE 410/510 - Codefest 7 | synth_top (compute_core)
Tool: Yosys 0.9 | Target: SKY130 HD

## Clock and Slack

The synthesis script targeted 200 MHz (5.0 ns clock period). Yosys completed
the synthesis pass but the LTP (longest path) analysis was blocked by combinational
loops in pipe_data[0], so no numeric slack value was produced. Worst-case slack
is therefore unknown from this run. The loop warnings indicate the S1 tready
combinational feedback (s_axis_tready depends on pipe_valid[0] which feeds back
into the input mux) was interpreted by Yosys as a combinational cycle.

## Critical Path

The timing path could not be fully resolved due to the detected loops in pipe_data[0].
Based on the cell breakdown, the dominant path runs through the S5 softmax normalize
stage: running_sum (24-bit) feeds 8 parallel $div cells, each performing a 24-bit
divide. Division is the deepest combinational operation in the design. The path is:
running_sum register -> $ne (zero check) -> $div x8 -> norm_beat mux -> pipe_data[4]
register. The $mul cells in S7 (Welford M2) are the second likely critical path
candidate.

## Cell Area and Top Contributors

Technology mapping to SKY130 standard cells did not complete because the SKY130
liberty file download failed in Colab and abc fell back to generic cells. The raw
pre-mapping statistics show 92 generic cells total:

  $adff  : 27  (pipeline registers across 8 stages)
  $mux   : 26  (stage select and tready back-pressure muxes)
  $div   :  8  (S5 softmax normalize, one per byte lane, unrolled)
  $mul   :  8  (S7 Welford M2, one per byte lane, unrolled)
  $add   :  8  (S4 running sum accumulation, unrolled)
  $gt    :  9  (S2 online max comparators + S3 LUT index selects)

The 8 $div cells are the most area-critical. On SKY130 HD, integer division is
not a primitive and abc would expand each $div into a multi-cycle divider or a
large combinational tree of adders and subtractors, likely dominating final area.

## Warnings and Issues

7,235 total warnings, all rooted in two causes. First, the three pipeline arrays
(pipe_data, pipe_valid, pipe_last) were declared as unpacked arrays of reg, which
Yosys 0.9 treats as memories and then replaces with individual registers, generating
one warning per replacement. Second, the combinational always block driving
s_axis_tready reads pipe_valid[0] while the clocked always block in S1 also writes
pipe_valid[0], creating a feedback loop that Yosys flags on every bit of pipe_data[0]
during the LTP pass (64 bits x repeated passes = bulk of the 7,235 count). These
are structural warnings, not functional errors, but the tready loop must be broken
before a clean STA run is possible.
