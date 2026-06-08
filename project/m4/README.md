# Project Milestone 4 - README
## ECE 410/510 HW4AI | Spring 2026
## Fused Softmax + LayerNorm Accelerator
## Author: Mahesh Naidu

---

## File Catalog

| File | Description | Checklist / Report Section |
|------|-------------|---------------------------|
| `rtl/top.sv` | Thin wrapper instantiating interface_mod. No glue logic. | §2 Source code |
| `rtl/compute_core.sv` | 8-stage fused pipeline: S1 latch, S2 max, S3 exp LUT, S4 sum, S5 softmax, S6 Welford mean, S7 Welford M2, S8 output. final_sum fix vs M3. | §2 Source code, Report §4 |
| `rtl/interface.sv` | AXI4-Lite register bank (0x00–0x10) + AXI4-Stream pass-through. Instantiates compute_core. | §2 Source code, Report §5 |
| `tb/tb_top.sv` | End-to-end testbench. Drives via AXI4-Lite config + AXI4-Stream data. Prints PASS/FAIL. | §2 Testbench |
| `sim/final_run.log` | Real iverilog 12.0 simulation output — RESULT: PASS 8/8 beats. | §2 Simulation log |
| `sim/final_waveform.png` | GTKWave screenshot showing 3 annotated regions. | §2 Waveform |
| `synth/config.json` | OpenLane 2 config: clock 10 ns, top module, SKY130 HD. | §3 Synthesis |
| `synth/openlane_run.log` | Yosys 0.9 full synthesis log. Exit 0. Hash 9c2bb123ba. | §3 Synthesis |
| `synth/timing_report.txt` | LTP depth 275, critical path final_sum→$div→pipe_data[4], ~9-13 ns. | §3 Synthesis, Report §7 |
| `synth/area_report.txt` | 132 cells by type. $div dominant (~65% area). Est. ~3,689 µm². | §3 Synthesis, Report §7 |
| `synth/power_report.txt` | Power estimation attempted. Est. ~58 µW dynamic at 100 MHz. | §3 Synthesis, Report §7 |
| `bench/benchmark.md` | Throughput + speedup vs SW baseline. 1,466× projected speedup. | §4 Benchmark |
| `bench/benchmark_data.csv` | Raw numbers — all reported values traceable to this file. | §4 Benchmark |
| `bench/roofline_final.png` | Roofline with SW measured point and HW projected point labeled. | §4 Benchmark, Report §8 |
| `report/design_justification.pdf` | 9-section design report, 2000-5000 words. | §5 Report |
| `report/figures/` | Block diagram, dataflow, roofline, waveform figures for report. | §5 Report |

**Diff from M3:**
- Module names changed: `compute_core_m3` → `compute_core`, `interface_mod_m3` → `interface_mod`
- `final_sum` register added to fix Verilog non-blocking assignment race in S4
- All other logic identical to M3

---

## Simulation Reproduction

```bash
iverilog -g2012 -o sim_m4 \
  project/m4/tb/tb_top.sv \
  project/m4/rtl/top.sv \
  project/m4/rtl/interface.sv \
  project/m4/rtl/compute_core.sv
vvp sim_m4
```
Expected: `RESULT: PASS`

## Synthesis Reproduction

Tool: Yosys 0.9 (Google Colab: `apt-get install yosys`)
Config: `project/m4/synth/config.json`
Clock: 10.0 ns (100 MHz)
