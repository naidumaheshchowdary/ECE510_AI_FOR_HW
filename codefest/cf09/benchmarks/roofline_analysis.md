# CF09 Roofline Analysis
## ECE 410/510 HW4AI | Spring 2026 | Codefest 09 CLLM

The accelerator point on the roofline is **projected**, not measured.
The throughput of 0.64 GOPS was derived from synthesis cycle counts
(8 cycles per row at 100 MHz target clock) rather than from a cocotb,
FPGA, or cycle-accurate simulation run.

The dominant uncertainty in this projection is **timing closure**. The
8 `$div` cells in Stage 5 create a combinational path of approximately
9–13 ns (confirmed by Yosys LTP, 275-node path, hash 40dbf23ba9), which
is marginal at 100 MHz and fails at 200 MHz. If post-place-and-route STA
on SKY130 shows the design closes at only 50 MHz, the projected sample
rate drops from 195,312 to 97,656 samples/sec and the speedup falls from
1,466× to 733×. The second uncertainty is AXI4-Stream back-pressure:
the projection assumes one beat accepted per cycle with no stalls. Any
downstream stall reduces effective throughput below the theoretical peak.

To convert this to a measured point, two steps are required: first,
complete full OpenLane 2 place-and-route on SKY130 to obtain the actual
achievable clock frequency; second, run an end-to-end cocotb simulation
with cycle-accurate timing to measure real throughput in simulation.
