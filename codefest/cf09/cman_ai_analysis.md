# CF09 CMAN - Arithmetic Intensity Analysis
## ECE 410/510 HW4AI | Spring 2026
## Mahesh Chowdary Naidu

---

## Task 1 - Dominant Kernel

Kernel    : Fused Softmax + LayerNorm
Operation : Numerically-stable streaming Softmax (S2-S5) followed by
            Welford online LayerNorm (S6-S8), fused into single pass
Input     : [B=8, T=64, d=64] x INT8 (hardware) / FP64 (SW baseline, NumPy)
Output    : [B=8, T=64, d=64] x INT8 (hardware) / FP64 (SW baseline)
d         : 64  - embedding dimension, 64 INT8 elements per token vector
T         : 64  - sequence length, 64 tokens per sequence
B         : 8   - batch size, 8 sequences per forward pass

One invocation (hardware) = one row of d=64 elements = 8 AXI4-Stream beats
                            of 8 bytes each, processed through all 8 stages

How I identified the kernel:

Looking at compute_core_m3.sv stage by stage, there is no matrix multiply
anywhere - no activation x weight multiplication, no GEMM. Instead:

  S2 tracks running_max - this is the numerically stable Softmax algorithm
  S3 looks up exp_lut - the exponential step of Softmax
  S4 accumulates running_sum - the denominator of Softmax
  S5 divides exp by sum - the normalization output of Softmax
  S6-S7 compute welford_mean and welford_m2 - Welford online algorithm for
        mean and variance, which is LayerNorm
  S8 outputs the normalized result - final LayerNorm output

So S2-S5 = Softmax, S6-S8 = LayerNorm, fused into one streaming pass.
These two operations together accounted for 28% of runtime in the M1 profiling.

---

## Task 2 - FLOPs Count

Note on S3: I counted exp_lut as 0 FLOPs. A LUT lookup is a memory read from
a ROM, not an arithmetic operation. Counting it would overstate actual compute.

Note on S4: This is not a simple per-element count. The RTL adds 8 bytes per
beat using an 8-way addition tree (8 adds), then accumulates beat_sum into
running_sum (1 more add) = 9 adds per beat, over 8 beats = 72 additions total.

Stage S2 - running max:
  Operations per element : 1 comparison
  Elements per row       : 64
  Subtotal               : 64 ops

Stage S3 - exp LUT:
  Operations per element : 0 (ROM read, not a FLOP)
  Elements per row       : 64
  Subtotal               : 0 ops

Stage S4 - running sum:
  Operations per beat    : 8 adds (byte tree) + 1 accumulate = 9
  Beats per row          : 8
  Subtotal               : 72 ops

Stage S5 - normalize:
  Operations per element : 1 multiply + 1 divide = 2
  Elements per row       : 64
  Subtotal               : 128 ops

Stage S6 - Welford mean:
  Operations per element : 1 subtract + 1 add + 1 shift = 3
  Elements per row       : 64
  Subtotal               : 192 ops

Stage S7 - Welford M2:
  Operations per element : 2 subtracts + 1 multiply + 1 add = 4
  Elements per row       : 64
  Subtotal               : 256 ops

Stage S8 - output:
  Operations per element : 0 (pass-through)
  Elements per row       : 64
  Subtotal               : 0 ops

Total FLOPs per row                   = 64 + 0 + 72 + 128 + 192 + 256 + 0
                                      = 712 FLOPs

Total FLOPs per invocation (B x T rows) = 712 x 64 x 8
                                        = 364,544 FLOPs

---

## Task 3 - Bytes Transferred

Reuse pattern : no-local-reuse (streaming)

The kernel has no weight matrix and no on-chip reuse. Each input element
is read exactly once, computed on, and each output element is written
exactly once. Data streams in and streams out with nothing held for reuse.

No-reuse lower bound (worst case - all data comes from off-chip):

  Formula : bytes = (d x bytes_per_element_in) + (d x bytes_per_element_out)
  Values  : (64 x 1) + (64 x 1)
  Bytes   : 128 bytes per row

Full-reuse upper bound (best case - input is cached on-chip):

  Formula : bytes = d x bytes_per_element_out
  Values  : 64 x 1
  Bytes   : 64 bytes per row

---

## Task 4 - Arithmetic Intensity

AI_low  = 712 / 128 =  5.56 FLOPs/byte
AI_high = 712 /  64 = 11.13 FLOPs/byte

Platform ridge point for SKY130 at 100 MHz:
  Peak compute : 1 GOPS
  Peak BW      : 0.8 GB/s  (AXI4-Stream 64-bit at 100 MHz)
  Ridge point  : 1.0 / 0.8 = 1.25 FLOPs/byte

Both AI_low (5.56) and AI_high (11.13) are greater than the ridge point (1.25).
Both bounds sit to the right of the ridge point on the roofline plot.
The kernel is compute-bound on SKY130.

Roofline sketch (log scale):

  GOPS
   1.0 |. . . . . . . .ridge. .x========compute ceiling===========
       |               point /
       |                   /
       |                 /
       |     BW ceiling/
  0.1  |             /
       |           /
       +----------+----------+----------+----------+--> FLOPs/byte
                 1.25       5.56       11.13
                 ridge      AI_low     AI_high
                 point      [----kernel range----]

---

## Task 5 - Bottleneck and Improvement

Is the design limited by interface BW, on-chip memory BW, or compute units?

The design is limited by compute units. Both arithmetic intensity bounds
(AI_low = 5.56 and AI_high = 11.13 FLOPs/byte) are above the SKY130 ridge
point of 1.25 FLOPs/byte, which means the kernel is compute-bound. The
AXI4-Stream interface can keep up with the data rate - the actual bottleneck
is the 8 division cells in Stage 5, which take roughly 8-12 ns each on SKY130
and are the reason the design cannot close timing at 200 MHz.

Single highest-leverage change:

Replace the 8 div cells on the critical path
(final_sum DFF -> div -> pipe_data[4] DFF) with a single reciprocal
computation (inv_sum = 1/running_sum) followed by 8 integer multiplications
(exp_val x inv_sum). This reduces the critical path delay from 8-12 ns down
to roughly 2-3 ns and would allow the design to close timing at 200 MHz.
