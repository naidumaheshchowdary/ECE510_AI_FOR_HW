# Heilmeier Catechism
## ECE 410/510 HW4AI | Spring 2026 | M1
## Project: Fused Softmax + Layer Normalization Accelerator

---

## Q1 — What are you trying to do?

Every transformer forward pass runs two normalization operations after
the attention block and after each feed-forward layer: softmax and layer
normalization. These operations appear in every transformer layer, on
every token, on every inference call. The algorithm being accelerated is
the professor's pure-NumPy transformer language model
(transformer_lm/transformer.py), which implements these operations
without any ML library.

This project builds a hardware accelerator chiplet in SystemVerilog that
computes softmax and layer normalization in a single streaming pass over
the input data. The chip reads each element once, computes both
normalization values on the fly using online algorithms, and writes the
result once — eliminating the repeated memory passes that make the
software implementation slow.

The target kernel is `softmax` (transformer.py line 16) and
`layer_norm_forward` (transformer.py line 84), which together account
for 28% of the forward pass runtime in the profiled baseline. The
chiplet uses an 8-stage pipelined datapath, an AXI4-Stream data
interface, and an AXI4-Lite control interface, and is synthesized with
OpenLane 2 on the SKY130 open-source PDK.

Measurable goal: reduce the combined softmax + layer normalization
latency from the measured software baseline to under 1 microsecond per
call, targeting a 3× or better overall speedup on the normalization
portion of the forward pass.

---

## Q2 — How is it done today and what are the limits?

The current software runs on a CPU using the professor's pure-NumPy
transformer code with no GPU or ML library. I measured the baseline
directly by running cProfile on train.py for 100 training steps.

Platform: Intel Core i7 laptop, Windows 11, Python 3.11, NumPy 1.26.
Model config: d=64, n_heads=4, d_ff=256, n_layers=2, seq_len=64,
batch_size=8. All weights are FP64.

Measured results (30-run median on the forward pass):

| Metric | Value |
|--------|-------|
| Median forward pass latency | 17.25 ms |
| Throughput | 6.38 GFLOP/s |
| Samples per second | 463.9 |
| Peak memory (RSS) | 4.25 MB |
| Total FLOPs per forward | 110,100,480 |

cProfile breakdown of the forward pass (100 steps, 2 layers):

| Function | Calls | Cumtime | % of forward |
|----------|-------|---------|--------------|
| ff_forward | 200 | 14.1 s | 60% |
| gelu | 200 | 9.8 s | 41% |
| mha_forward | 200 | 7.7 s | 32% |
| softmax | 300 | 5.1 s | 21% |
| layer_norm_forward | 500 | 1.5 s | 7% |

Softmax and layer normalization together take 6.6 seconds out of
23.7 seconds of forward pass time across 100 training steps, which
is 28% of total forward cost.

The core problem is memory traffic. Softmax requires three passes over
the data: one to find the maximum, one to compute exponentials, one to
divide. Layer normalization requires three more: mean, variance,
normalize. That is six memory reads of the same T×d matrix for only
13×d×B floating-point operations per row. The arithmetic intensity for
this unfused pipeline is:

  AI = (B×T×13×d FLOPs) / (B×T×d×8 bytes × 6 passes)
     = 13 / 48
     = 0.271 FLOP/byte

This is well below the CPU's ridge point of 0.625 FLOP/byte (FP64),
placing the kernel firmly in the memory-bound region of the roofline.
The CPU spends most of its time waiting for data rather than computing.

---

## Q3 — What is new and why will it succeed?

The key idea is operator fusion: instead of six memory passes, read
each element exactly once.

This is done using the Welford online algorithm, which updates the
running mean and variance one element at a time as data streams through,
and the online-safe softmax algorithm, which maintains a running maximum
and running sum simultaneously. Together these eliminate all intermediate
storage and reduce memory traffic from six passes to one.

The arithmetic intensity of the fused kernel:

  AI fused (FP64) = 13d×B×T / (d×B×T×8×2) = 0.813 FLOP/byte

This is above the CPU's FP64 ridge point of 0.625, moving the kernel
from memory-bound to compute-bound. With INT8 quantization:

  AI fused (INT8) = 13d×B×T / (d×B×T×1×2) = 6.500 FLOP/byte

This places the kernel firmly in the compute-bound region of the
accelerator's roofline (hardware ridge = 0.256 FLOP/byte at 102.4
GOPS/s with 400 GB/s on-chip SRAM bandwidth).

The hardware is an 8-stage pipeline: input latch → online max → exp
LUT → running sum → normalize → Welford mean → Welford variance →
layer norm output. Each stage processes one element per clock cycle.
At 200 MHz the full T×d = 64×64 = 4096-element result is produced in
4096 + 7 = 4103 cycles = 20.5 microseconds — compared to the per-call
software latency of roughly 17,000 µs / (200 steps × 2 layers) ≈
42 µs per combined softmax+layernorm call.

This approach will succeed for three reasons. First, the hardware is
a pipeline of simple arithmetic units, not a complex 2D array — timing
closure in SKY130 with OpenLane 2 is realistic. Second, the exp
approximation using an 8-entry LUT with linear interpolation gives less
than 0.1% error, which is acceptable for inference normalization.
Third, the design is scoped to fixed dimensions matching the professor's
transformer config (d=64, T=64), so no tiling complexity is needed.
