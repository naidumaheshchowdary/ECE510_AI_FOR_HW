# Milestone 2 — How to Reproduce Simulation
## ECE 410/510 HW4AI | Spring 2026
## Project: Fused Softmax + Layer Normalization Accelerator

---

## Simulator

**Icarus Verilog 12.0** (`iverilog` + `vvp`)

Install on Windows (WSL2 recommended) or Linux:
```bash
sudo apt-get install iverilog   # Ubuntu/Debian
```
Or download the Windows native installer from:
https://bleyer.org/icarus/

Verify version:
```
iverilog -V
```
Expected: `Icarus Verilog version 12.0`

---

## Repository Structure (M2)

```
project/
├── m2/
│   ├── rtl/
│   │   ├── compute_core.sv      ← compute core (top module: compute_core)
│   │   └── interface.sv         ← interface wrapper (top module: interface_mod)
│   ├── tb/
│   │   ├── tb_compute_core.sv   ← compute core testbench
│   │   └── tb_interface.sv      ← interface testbench
│   ├── sim/
│   │   ├── compute_core_run.log ← simulation transcript (committed)
│   │   ├── interface_run.log    ← simulation transcript (committed)
│   │   └── waveform.png         ← representative waveform
│   ├── precision.md             ← numerical format + error analysis
│   └── README.md                ← this file
```

---

## Running the Simulations

Run both commands from the `project/` directory.

### 1. Compute Core Testbench

```bash
iverilog -g2012 -o sim_core \
    m2/tb/tb_compute_core.sv \
    m2/rtl/compute_core.sv

vvp sim_core | tee m2/sim/compute_core_run.log
```

**Expected output (last line):**
```
PASS: tb_compute_core
```

### 2. Interface Testbench

```bash
iverilog -g2012 -o sim_if \
    m2/tb/tb_interface.sv \
    m2/rtl/interface.sv \
    m2/rtl/compute_core.sv

vvp sim_if | tee m2/sim/interface_run.log
```

**Expected output (last line):**
```
PASS: tb_interface
```

---

## Windows Native (without WSL)

If running Icarus on Windows natively (not WSL), use backslashes and
cmd.exe or PowerShell:

```cmd
iverilog -g2012 -o sim_core m2\tb\tb_compute_core.sv m2\rtl\compute_core.sv
vvp sim_core
```

```cmd
iverilog -g2012 -o sim_if m2\tb\tb_interface.sv m2\rtl\interface.sv m2\rtl\compute_core.sv
vvp sim_if
```

---

## Known Issues Fixed in M2

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `interface.sv:55: syntax error` | `module interface` — `interface` is a reserved keyword in SystemVerilog | Module renamed to `interface_mod` in both `interface.sv` and `tb_interface.sv` |
| `compute_core.sv:151: sorry: constant selects in always_* processes not supported` | Variable part-select `pipe_data[1][b*8+2 -: 3]` inside `always_comb` is unsupported in Icarus 12 | Loop fully unrolled; each of the 8 bytes extracted as explicit named signals (`byte0`..`byte7`) |
| `logic` keyword in port declarations | `logic` in port I/O positions can cause warnings in pure Verilog-2005 mode | Changed port types to `reg`/`wire` as appropriate; `always_ff`/`always_comb` replaced with `always @(posedge clk...)` / `always @(*)` |

---

## Deviations from M1 Plan

None. The interface selection (AXI4-Lite control + AXI4-Stream 64-bit
data), precision choice (INT8), and pipeline depth (8 stages) are
unchanged from M1.

The `interface_mod` naming is a purely cosmetic fix required by
SystemVerilog/Verilog language rules and does not affect any protocol
behavior or register map.

---

## Dependencies

No Python pre/post-processing is required to run the simulations.
The testbenches are self-contained and print PASS/FAIL directly.

If you wish to reproduce the quantization error analysis in
`precision.md`, you need:
- Python 3.11+
- NumPy 1.26+

No additional packages beyond the standard library are required.
