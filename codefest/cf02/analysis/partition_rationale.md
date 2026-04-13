
# HW/SW Partition Rationale
**Project:** Fused Softmax + LayerNorm Accelerator | **ECE 410/510 Spring 2026**

---

## (a) Which kernels to accelerate and why the roofline supports it

The two functions selected for hardware acceleration are `transformer.py:softmax` and `transformer.py:layer_norm_forward`. Together they consumed **6.824 seconds** out of the 65.347-second training run (transformer.py:forward + backward combined), accounting for roughly **18% of the total forward-pass compute time** across 100 training steps. Both functions are called hundreds of times per run (300 and 500 calls respectively), making them high-frequency targets.

The roofline analysis confirms this choice. Running on an Intel i5-10210U (peak ≈ 46.4 GFLOP/s, DRAM bandwidth ≈ 34.1 GB/s, ridge point ≈ 1.36 FLOP/byte), the unfused softmax sits at **AI = 0.156 FLOP/byte** and layer_norm_forward at **AI = 0.200 FLOP/byte** - both deep in the memory-bound region, far left of the ridge point. This means the CPU spends most of its time waiting for data, not doing arithmetic. A fused hardware pipeline that streams each element once and computes softmax normalization and layer norm in a single pass raises the effective AI to approximately **0.367 FLOP/byte**, nearly tripling data reuse without changing the algorithm.

## (b) What the software baseline continues to handle

The host CPU retains responsibility for the full training loop: data loading and batching, embedding lookup, the Q/K/V projection matrix multiplications inside `mha_forward`, the feed-forward GELU layers (`ff_forward`/`ff_backward`), the Adam optimizer step, loss computation, and all backpropagation passes. The backward passes for softmax and layer_norm (`layer_norm_backward`) are also left in software for this milestone, as the forward-pass acceleration alone captures the majority of the inference-time benefit.

## (c) Interface bandwidth requirement

The accelerator processes one row of the attention matrix per invocation. For the transformer_lm config (d_model = 64, batch = 8, seq_len = 64), each softmax call processes a vector of 64 FP64 values (512 bytes in, 512 bytes out = 1,024 bytes per call). At 300 calls per 100 steps and a target throughput matching the CPU baseline rate, the required interface bandwidth is approximately **0.307 MB per 100 steps**, or well under **1 MB/s** sustained. An AXI4-Stream interface at even 32-bit width and 100 MHz delivers ~400 MB/s — more than 1,000× the needed bandwidth. The design is therefore **not interface-bound** at the target throughput; the bottleneck remains on-chip compute and the quality of the fused pipeline implementation.

## (d) Bound classification and expected change

On the current CPU hardware, both kernels are **memory-bound** (AI ≪ ridge point of 1.36 FLOP/byte). The accelerator design targets a custom on-chip SRAM pipeline with a peak bandwidth of 512 GB/s and a target compute of 200 GOPS, placing the ridge point at
approximately 0.39 FLOP/byte. Since the fused kernel's AI of 0.367 FLOP/byte sits just below this ridge, the accelerator design moves the kernel from deeply memory-bound (CPU) to **near the ridge point** - balanced between compute and memory, which is the optimal operating region for a streaming pipeline accelerator.
