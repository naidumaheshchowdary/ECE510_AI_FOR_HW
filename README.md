# ECE510: Hardware for Artificial Intelligence and Machine Learning 

## Project Title : Fused Softmax + LayerNorm Accelerator
## **Author: Mahesh Naidu| Spring 2026 | Portland State University| Prof. Christof Teuscher**

This repository contains the complete Milestone 4 submission for the Hardware
for Artificial Intelligence and Machine Learning course (ECE 410/510).

---

## Project Summary

A synthesizable, verified hardware accelerator for fused Softmax + LayerNorm
targeting the SKY130 open-source PDK. The accelerator implements an 8-stage
AXI4-Stream pipeline that processes INT8 activations from a transformer language
model, achieving a projected 1,466× throughput improvement over the NumPy CPU
baseline (133.2 → 195,312 samples/sec).

---

## M4 Submission

**M4 folder:** [`project/m4/`](project/m4/)

**Design justification report:** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)

**M4 README (full file catalog):** [`project/m4/README.md`](project/m4/README.md)

**Simulation result:** PASS 8/8 beats — [`project/m4/sim/final_run.log`](project/m4/sim/final_run.log)

**Synthesis:** 132 cells, LTP=275, hash 9c2bb123ba — [`project/m4/synth/`](project/m4/synth/)

---

## Earlier Milestones

- M1: [`project/m1/`](project/m1/) — profiling, kernel selection, interface decision
- M2: [`project/m2/`](project/m2/) — RTL, unit verification
- M3: [`project/m3/`](project/m3/) — integration, co-simulation, synthesis

---

## Quick Start

```bash
# Simulate
iverilog -g2012 -o sim_m4 project/m4/tb/tb_top.sv \
  project/m4/rtl/top.sv project/m4/rtl/interface.sv \
  project/m4/rtl/compute_core.sv && vvp sim_m4
```
