# Heilmeier Questions
**Project:** Fused Softmax + LayerNorm Accelerator | **ECE 410/510 Spring 2026**

---

## Q1: What are you trying to do?

Every transformer model — the kind that powers language translation, text prediction, and speech recognition — does a specific cleanup step after each attention calculation. This step involves two operations called softmax and layer normalization. Both operations read every number in a data array, do a small amount of arithmetic, then read the array again one or more times to finish the calculation. This back-and-forth is wasteful: the processor spends most of its time waiting for data to arrive from memory rather than actually computing anything useful.

The goal of this project is to build a small custom hardware chip (described in SystemVerilog and synthesized using the open-source OpenLane 2 tool) that reads each number exactly once and computes both softmax and layer normalization in a single streaming pass. The target outcome is a measurable reduction in the time spent on these two operations - from a combined 6.8 seconds per 100 training steps on a laptop CPU down to under 0.1 seconds — using a chip that connects to the host processor through a standard AXI4-Stream data interface.

---

## Q2: How is it done today, and what are the limits of current practice?

**Measured software baseline** (Intel i5-10210U, Python 3.11, FP64, 100 training steps on the professor's `transformer_lm` code):

| Function | Calls | Cumtime (s) | % of train() |
|---|---|---|---|
| `transformer.py:softmax` | 300 | 5.198 | ~8% |
| `transformer.py:layer_norm_forward` | 500 | 1.626 | ~2.5% |
| Combined | — | **6.824** | **~10.4%** |

The current approach runs both functions sequentially on a general-purpose CPU using NumPy. Each function makes multiple passes over the same data: softmax requires a max-subtraction pass, an exp pass, a sum pass, and a division pass — four total reads
of the input vector. Layer normalization similarly requires a mean pass, a variance pass, and a normalization pass. On the i5-10210U with 34.1 GB/s DRAM bandwidth and a ridge point of 1.36 FLOP/byte, the softmax arithmetic intensity is only **0.156 FLOP/byte**
and layer norm is **0.200 FLOP/byte**. Both sit far to the left of the ridge point, meaning the CPU is almost entirely idle waiting for memory reads to complete.

The fundamental limit is that a general-purpose CPU cannot eliminate these redundant memory passes because its instruction pipeline was not designed for single-pass streaming normalization. No open-source, synthesizable (OpenLane 2 / SKY130 compatible) fused
accelerator for these two operations currently exists.

---

## Q3: What is new in your approach and why will it succeed?

**What changed after profiling:** The profiling confirmed that softmax is the dominant forward-pass bottleneck at 8% of total runtime, but also revealed that `ff_forward` (GELU feed-forward, 26% of runtime) and `mha_forward` (matrix multiplications, 14%) are individually larger. However, those kernels involve large matrix multiplications that are harder to pipeline cleanly in a solo synthesizable design. The softmax + layernorm pair offers a better tradeoff: high call frequency (800 combined calls per 100 steps), pure streaming structure with no matrix dimensions, and a clean single-pass fusion opportunity.

**The new approach** uses a Welford online algorithm to compute both softmax and layer norm statistics (mean, variance, running sum) in a single forward pass over the data. Instead of four separate memory reads, the hardware pipeline reads each element exactly once and produces the normalized output directly. This raises the effective arithmetic intensity from 0.156 F/B (softmax unfused) to approximately **0.367 F/B** (fused pipeline) — a 2.4× improvement purely from eliminating redundant memory traffic, with no change to the mathematical result.

The design will succeed for three concrete reasons. First, the hardware is a simple 8-stage pipeline with no tiling or 2D spatial indexing — every stage processes one element per clock cycle, making timing closure straightforward in OpenLane 2 on SKY130.
Second, the fused kernel's AI of 0.367 FLOP/byte sits near the ridge point of the target accelerator (0.39 FLOP/byte at 200 GOPS / 512 GB/s on-chip bandwidth), meaning the design is balanced and the compute utilization will be high. Third, the AXI4-Stream interface bandwidth needed is under 1 MB/s sustained — well within the capability of even a narrow 32-bit interface at 100 MHz — so the design will not be interface-bound.
