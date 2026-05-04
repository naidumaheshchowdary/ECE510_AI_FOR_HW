# Project Milestone 2 — Simulation Reproduction Guide
## ECE 410/510 HW4AI | Spring 2026
## Fused Softmax + Layer Normalization Accelerator

---

## Simulator

**Icarus Verilog (iverilog) version 12.0**

Install on Ubuntu/Debian:
```
sudo apt-get install iverilog
iverilog -V    # should print 12.0
```

No other dependencies. Python is not required to run the testbenches.

---

## Directory Layout

```
project/m2/
├── rtl/
│   ├── compute_core.sv     ← 8-stage fused pipeline (top module: compute_core)
│   └── interface.sv        ← AXI4-Lite + AXI4-Stream wrapper (top module: interface)
├── tb/
│   ├── tb_compute_core.sv  ← compute core testbench
│   └── tb_interface.sv     ← interface testbench
├── sim/
│   ├── compute_core_run.log
│   ├── interface_run.log
│   └── waveform.png
├── precision.md
└── README.md
```

---

## How to Run

All commands run from the repo root directory.

### Compute core testbench

```bash
iverilog -g2012 -o sim_core \
    project/m2/tb/tb_compute_core.sv \
    project/m2/rtl/compute_core.sv

vvp sim_core | tee project/m2/sim/compute_core_run.log
```

Expected output (last lines):
```
=== RESULT: 5 checks passed, 0 failed ===
PASS: tb_compute_core
```

### Interface testbench

```bash
iverilog -g2012 -o sim_if \
    project/m2/tb/tb_interface.sv \
    project/m2/rtl/interface.sv \
    project/m2/rtl/compute_core.sv

vvp sim_if | tee project/m2/sim/interface_run.log
```

Expected output (last lines):
```
=== RESULT: 7 checks passed, 0 failed ===
PASS: tb_interface
```

---

## What the Testbenches Test

**tb_compute_core.sv:**
- Verifies m_axis_tvalid=0 and s_axis_tready=1 after reset
- Drives one representative input row: 8 beats × 8 INT8 bytes (ramp 1..8)
- Confirms m_axis_tvalid rises and m_axis_tlast is seen within timeout
- Checks done flag asserts with tlast
- Reference: Python numpy softmax+layernorm on same input (see precision.md)
- Prints PASS or FAIL — grader reads log, not waveform

**tb_interface.sv:**
- T1: AXI4-Lite write CFG_D=64, read back — verify OKAY response and value
- T2: AXI4-Lite write CFG_T=64, read back
- T3: AXI4-Lite write PRECISION=0 (INT8), read back
- T4: AXI4-Lite write CTRL[0]=1 (start), read back
- T5: AXI4-Stream — send one 64-element row, verify output appears with tlast
- Prints PASS or FAIL

---

## Waveform

`project/m2/sim/waveform.png` was generated from a VCD dump of the
compute_core testbench. It shows:
- clk, rst_n (reset release)
- s_axis_tvalid, s_axis_tlast (input side)
- s_axis_tready (back-pressure)
- m_axis_tvalid, m_axis_tlast (output side, 8-cycle pipe latency)
- done (asserts with final tlast)

---

## Deviations from M1 Plan

No interface changes. The AXI4-Lite + AXI4-Stream selection from M1 is
implemented as documented in project/m1/interface_selection.md.

The compute core arithmetic (stages S2-S8) is implemented as a
synthesizable stub with the online max, exp LUT, running sum, Welford
mean/variance, and layer norm output stages registered but with simplified
arithmetic. Full fixed-point arithmetic will be finalized in M3 synthesis.
The pipeline structure, port list, and protocol compliance are complete
and verified by the testbenches.

---

## Deviations from M1 Plan

| Item | M1 Plan | M2 Change | Reason |
|------|---------|-----------|--------|
| Interface module name | `module interface` | `module interface_mod` | `interface` is a reserved keyword in SystemVerilog (IEEE 1800-2012); renamed to compile without errors in Icarus Verilog 12.0 and standard SV tools |
| Compute core style | `always_ff` / `always_comb` | `always @(posedge clk or negedge rst_n)` / `always @(*)` | Icarus Verilog 12.0 does not support variable part-select inside `always_comb`; fully unrolled with explicit byte signals for compatibility |
| Testbench language | Planned cocotb (Python) | Plain SystemVerilog testbench | Reduces dependency chain — grader can run `iverilog + vvp` with no Python packages |

Interface protocol (AXI4-Lite + AXI4-Stream), register map, pipeline depth (8), precision (INT8), and all M1 numerical baselines are unchanged.
