# Remaining Tasks Before M4
## ECE 410/510 HW4AI | Spring 2026

1. **Replace 8× `$div` cells in `compute_core_m3.sv` Stage 5 with
   reciprocal multiply to eliminate the 9–13 ns critical path**
   The path `final_sum DFF ($procdff$579) → $ne → 8×$div → pipe_data[4]
   DFF ($procdff$575)` is the bottleneck (LTP node 269–272, hash
   40dbf23ba9). Replace `({16'd0,eb} * 24'd255) / final_sum` with
   `(eb * (16'hFFFF / final_sum[15:0])) >> 8` using a pre-computed
   reciprocal register. Expected result: eliminates all 8 `$div` cells
   (~2,400 µm² estimated), reduces critical path from ~9–13 ns to
   ~2.5 ns, enables 200 MHz timing closure, doubles projected throughput
   from 195,312 to 390,625 samples/sec.

2. **Fix signed/unsigned width mismatches in Welford S6/S7 stages to
   eliminate 649 unique synthesis warnings (91,944 total)**
   Lines 157–185 of `compute_core_m3.sv` mix `$signed({16'd0,
   pipe_data[4][7:0]})` with `$signed(welford_mean[23:0])` without
   explicit width matching. Add `$signed(24'(pipe_data[4][7:0]))`
   zero-extension casts at each Welford update to produce a clean
   synthesis log with zero width-mismatch warnings and remove ambiguity
   in the synthesized adder widths.

3. **Run full OpenLane 2 place-and-route on PSU research computing
   cluster to obtain mapped cell area (µm²), real setup slack from
   OpenSTA, and dynamic power from OpenROAD VCD annotation**
   Current `area_report.txt` shows `Chip area = 0.000000 µm²` because
   Yosys 0.9 on Colab uses generic ABC gates without SKY130 liberty.
   PSU cluster provides >8 GB RAM and full OpenLane 2 Docker image.
   Running `flow.tcl -design top -tag m4_run` will produce:
   `reports/synthesis/1-synthesis.stat` (real cell area),
   `reports/signoff/sta.rpt` (worst negative slack with SKY130 liberty),
   and `reports/signoff/power.rpt` (dynamic power with VCD activity) —
   all three required for complete M4 benchmark and report sections 7–8.
