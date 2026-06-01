# CF09 CMAN — Arithmetic Intensity Analysis
## ECE 410/510 HW4AI | Spring 2026
## Mahesh Chowdary Naidu

---

## Task 1 — Dominant Kernel

Kernel    : Fused Softmax + LayerNorm
Operation : Numerically-stable streaming Softmax (S2–S5) followed by
            Welford online LayerNorm (S6–S8), fused into single pass
Input     : [B=8, T=64, d=64] × INT8 (hardware) / FP64 (SW baseline, NumPy)
Output    : [B=8, T=64, d=64] × INT8 (hardware) / FP64 (SW baseline)
d         : 64  — embedding dimension, 64 INT8 elements per token vector
T         : 64  — sequence length, 64 tokens per sequence
B         : 8   — batch size, 8 sequences per forward pass

One invocation (hardware) = one row of d=64 elements = 8 AXI4-Stream beats
                            of 8 bytes each, processed through all 8 stages
---

## Task 2 — FLOPs Count

Stage S2 (running max):
  Operations per element : 
  Elements per row       : 
  Subtotal               : 

Stage S3 (exp LUT):
  ...

Stage S4 (running sum):
  ...

Stage S5 (normalize):
  ...

Stage S6 (Welford mean):
  ...

Stage S7 (Welford M2):
  ...

Stage S8 (output):
  ...

Total FLOPs per row = 
Total FLOPs per invocation (all B×T rows) = 

---

## Task 3 — Bytes Transferred

Reuse pattern : [name it — streaming? no-local-reuse? output-stationary?]

No-reuse (lower bound):
  Formula : 
  Values  : 
  Bytes   : 

Full-reuse (upper bound):
  Formula : 
  Values  : 
  Bytes   : 

---

## Task 4 — Arithmetic Intensity

AI_low  = FLOPs / bytes_no_reuse  = 
AI_high = FLOPs / bytes_full_reuse = 

---

## Task 5 — Bottleneck and Improvement

Is design limited by: interface BW / on-chip memory BW / compute units?
Answer: 

Single highest-leverage change:
Answer:
