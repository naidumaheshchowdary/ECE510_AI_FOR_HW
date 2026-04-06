# Heilmeier Questions

## Q1: What are you trying to do?
```
This project designs, implements, and benchmarks a custom INT8 convolutional inference accelerator chiplet described in SystemVerilog, targeting the 3×3 Conv2D kernel used in the layer1 blocks of ResNet-style CNNs (64 input channels → 64 output channels, 56×56 spatial resolution). The chiplet receives INT8-quantized weights and activations from a host SoC through an AXI4-Stream interface, computes a full output feature map using a weight-stationary dataflow backed by a 36 KB on-chip weight SRAM, and returns INT32 partial sums to the host. The chiplet is synthesized using OpenLane 2 targeting the SKY130 open-source PDK. The goal is to deliver a measurable latency reduction over a CPU software baseline (measured: 4.12 ms per inference) while consuming less energy per inference than a general-purpose processor core.
```
## Q2: How is it done today and what are the limits?
```
Edge CNN inference today runs on one of three platforms:

ARM Cortex-A CPU with NEON SIMD (e.g., Raspberry Pi, Zynq PS): Achieves ~50–200 GOPS/s at INT8 via TFLite or ONNX Runtime. Measured
software baseline on this project's target kernel (3×3 Conv2D, 64ch, 56×56, FP32): 4.12 ms, 56.15 GFLOP/s. The CPU is compute-bound at
this kernel's arithmetic intensity (AI = 131.89 FLOP/byte) but wastes energy on instruction fetch, decode, and branch logic that contributes zero MAC throughput.

Mobile GPU (Mali, Adreno): Higher throughput but 5–10× higher power draw and significant area cost not viable for MCU-class edge devices. Off-shelf NPUs (Google Edge TPU, NXP eIQ, Kendryte K210): Fixed architectures with no customization, vendor lock-in, and closed RTL cannot be synthesized or modified for a custom dataflow or precision target.

Limits of current practice:

No open-source, synthesizable (OpenLane 2 compatible) INT8 conv accelerator with parameterized AXI4 interface exists at the SKY130 PDK level. General-purpose datapaths waste 40–60% of dynamic power on control logic that is irrelevant to the inner MAC loop. INT8 quantization reduces weight memory by 4× (144 KB → 36 KB) and raises arithmetic intensity from 131.89 to 144.00 FLOP/byte with on-chip weight reuse but CPUs cannot exploit this fully because their SIMD units were not designed for sustained INT8×INT8→INT32 accumulation at this tile size.
```
## Q3: What is new in your approach and why will it succeed?
```
This project applies a roofline-first hardware sizing methodology:
Arithmetic-intensity-driven SRAM sizing: The on-chip weight SRAM is sized exactly to hold one full layer's INT8 weight tensor (36 KB), eliminating all weight DRAM traffic during inference. This raises the effective AI from 140.77 (no reuse) to 144.00 FLOP/byte (full reuse), well above the memory bandwidth ridge point of the target interface. INT8 weight-stationary MAC array: A 16×16 array of INT8 multiply- accumulate units with INT32 accumulators delivers 256 MACs/cycle. At 200 MHz: 256 × 2 × 200M = 102.4 GOPS/s — a 1.82× improvement over the 56.15 GFLOP/s software baseline, with further gains from eliminating non-MAC CPU overhead (instruction decode, cache miss stalls). Standard AXI4-Lite + AXI4-Stream interface: Connects to any ARM AMBA SoC without custom glue logic. Bandwidth analysis confirms the design is balanced (interface time ≈ compute time) and not interface-bound. Fully synthesizable with OpenLane 2 / SKY130: Unlike GPU or NPU solutions, this design produces a GDSII layout, enabling area and power analysis that validates the roofline predictions at silicon level.
```
