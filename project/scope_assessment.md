# Project Scope Assessment
ECE 410/510 - Spring 2026 | Mahesh Chowdary Naidu
Project: Fused Softmax + Layer Normalization Accelerator
Updated: CF07

## Current Scope (confirmed)

8-stage pipelined compute core for fused INT8 softmax and layer normalization.
Target config: d=64, T=64, batch=8 on SKY130 HD at 100-200 MHz.
AXI4-Stream input and output, asynchronous active-low reset, single clock domain.

## What the CF07 Synthesis Revealed

The RTL elaborates cleanly (92 cells, 1831 wire bits, exit code 0) confirming the pipeline structure is logically sound. Two scope-relevant findings came out
of this run.

The 8 $div cells in S5 are not synthesizable as-is on SKY130. This affects the softmax normalize stage which is core to the M2 deliverable. Replacing them with a reciprocal LUT is a small RTL change but it is required before M3 and is now the top priority.

The S8 LayerNorm output is currently a stub (g=1, b=0, passthrough). Full scale and shift arithmetic is deferred to M3 as planned. This remains on scope but is explicitly not started yet.

## Scope Decision

Core scope is unchanged. The fused softmax + layernorm pipeline in INT8 at d=64 is achievable on SKY130 within M3 timeline once the divide fix is applied. FP64 precision (precision=1 port) remains a stretch goal and will not be attempted for M3. AXI4-Lite config interface integration is also deferred post-M3 if time allows.

