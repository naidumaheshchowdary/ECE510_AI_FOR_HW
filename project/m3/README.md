
# Project Milestone 3 — README
## ECE 410/510 HW4AI | Spring 2026
## Fused Softmax + Layer Normalization Accelerator

---

## File Catalog

| File | Description |
|------|-------------|
| `rtl/top.sv` | Integrated top module instantiating `interface_mod` (M2) and `compute_core` (M2) with full AXI4-Lite + AXI4-Stream port connections. No glue logic required. |
| `tb/tb_top.sv` | End-to-end co-simulation testbench. Drives design exclusively through AXI4-Lite config writes and AXI4-Stream data beats. Uses d=64 input vector (8 beats × 8 bytes). Prints PASS/FAIL. |
| `sim/cosim_run.log` | Co-simulation transcript showing PASS for all 8 output beats. Produced by iverilog 12.0 + vvp. |
| `sim/cosim_waveform.png` | Annotated waveform showing (1) AXI4-Lite host write region, (2) AXI4-Stream compute activity region, (3) AXI4-Stream host read region. |
| `synth/config.json` | OpenLane 2 synthesis configuration. Clock period 10.0 ns (100 MHz). Design name: top. Source files: top.sv, interface.sv, compute_core.sv. |
| `synth/openlane_run.log` | Full Yosys 0.9 synthesis stdout/stderr. Exit code 0. 89 cells, 129 wires. Chip area 0 (SKY130 liberty not mapped in Yosys 0.9). |
| `synth/timing_report.txt` | LTP-based timing report. Critical path: running_sum DFF → 8×$div → pipe_data[4] DFF, estimated 9–13 ns. Marginal at 100 MHz; fails at 200 MHz. |
| `synth/area_report.txt` | Cell count by type (89 total), wire count, estimated die area 3,000–5,000 µm². $div cells dominate at ~66% of area. |
| `synth/critical_path.md` | Critical path identification: start=running_sum DFF, end=pipe_data[4] DFF, bottleneck=8×$div. Explains why and how to fix (reciprocal multiply for M4). |
| `synth/power_report.txt` | Power estimation attempt. Full OpenLane 2 flow not available (4 GB RAM, need >8 GB). Manual estimate: ~44 µW dynamic at 100 MHz. M4 plan documented. |
| `synthesis_notes.md` | 914-word narrative: what synthesized, what failed, why, scope adjustment rationale, M4 recovery plan. |

---

## Co-Simulation Reproduction

**Simulator:** Icarus Verilog 12.0 (`iverilog --version` → `Icarus Verilog version 12.0`)

**Commands (run from repo root):**
```bash
# Compile
iverilog -g2012 -o sim_m3 \
  project/m3/tb/tb_top.sv \
  project/m3/rtl/top.sv \
  project/m2/rtl/interface.sv \
  project/m2/rtl/compute_core.sv

# Run simulation
vvp sim_m3

# View waveform (optional)
gtkwave m3_cosim.vcd
```

**Dependencies:** iverilog 12.0, vvp (included with iverilog), gtkwave (optional)

**Expected output:**
```
RESULT: PASS
```

---

## Synthesis Reproduction

**Tool:** Yosys 0.9 (Google Colab: `apt-get install yosys`)

**Commands:**
```python
# In Google Colab
import subprocess
script = '\n'.join([
    'read_verilog -sv /content/top.sv /content/interface.sv /content/compute_core.sv',
    'hierarchy -check -top top',
    'proc', 'flatten', 'opt', 'memory -nomap', 'opt',
    'dfflibmap -liberty /content/sky130_hd.lib',
    'abc -liberty /content/sky130_hd.lib',
    'opt_clean',
    'stat',
    'write_verilog -noattr /content/top_netlist.v',
])
with open('/content/synth.ys', 'w') as f: f.write(script)
subprocess.run('yosys /content/synth.ys', shell=True)
```

**OpenLane 2 version:** v2.0.0 (targeted; full flow requires >8 GB RAM)
**Config file:** `project/m3/synth/config.json`
**Environment:** SKY130 PDK, Docker image `efabless/openlane:v2.0.0`

Full OpenLane 2 Docker flow planned for M4 on PSU research computing cluster.

---

## Scope Adjustment Summary

Original clock target: 200 MHz (5.0 ns). Synthesis revealed 8× `$div` cells
in Stage 5 creating a ~9–13 ns critical path. M3 clock relaxed to 100 MHz (10.0 ns).
M4 will replace `$div` with reciprocal multiply to achieve 200 MHz closure.
All other scope unchanged: 8-stage pipeline, AXI4 interfaces, INT8, d=64/T=64.
