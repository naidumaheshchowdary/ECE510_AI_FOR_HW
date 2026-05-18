
# M3 Plan
ECE 410/510 - Codefest 7 | synth_top (compute_core)

## What I will fix before M3

Fix 1 - Break the tready combinational loop (highest priority).
The s_axis_tready assignment reads pipe_valid[0] inside a combinational block that feeds back into the S1 clocked block. I will register tready one cycle earlier using pipe_valid[1] instead, which eliminates the loop and unblocks the LTP timing analysis entirely.

Fix 2 - Replace $div with multiply-by-reciprocal.
The 8 unrolled $div cells in S5 are not mappable to SKY130 primitives and will explode into large combinational trees. Since running_sum is bounded to 8 x 255 = 2040 max, I will precompute a small reciprocal LUT (11 entries, one per possible sum) and replace each divide with a LUT lookup and a multiply, dropping the 8 $div cells entirely and making the path synthesizable.

Fix 3 - Obtain a working SKY130 liberty file.
I will pull the liberty file from the official google/skywater-pdk release on GitHub rather than the mirror that failed, so abc can complete technology mapping and produce real area and slack numbers for M3.

Target clock for M3: start at 5 ns (200 MHz) and tighten to 4 ns if slack allows after the divide fix.
