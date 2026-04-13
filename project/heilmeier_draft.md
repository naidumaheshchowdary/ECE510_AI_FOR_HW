# Heilmeier Questions

## Q1: What are you trying to do?
```
Every time a transformer model processes a word or token, it runs two normalization steps back to back: softmax and layer normalization. These are not the exciting parts of the model - they are the cleanup work that happens after the attention computation and after the feed-forwardlayer. But they run on every single token, every single layer, every single inference call. 

The problem is that today's software runs these two operations separately, which forces the processor to read the same data from memory six times just to normalize one row of numbers. That is wasteful. 

My project builds a small custom chip that does both operations in asingle pass it reads each number once, computes everything it needs on the fly, and writes the result once. The hardware pipeline runs at 200 MHz and finishes the job in a fraction of the time the software takes.

The target is a 3× or better speedup for the normalization step, using a design small enough to synthesize on the open-source SKY130 chip process with the OpenLane 2 tool.
```
## Q2: How is it done today and what are the limits?
```
Right now, this computation runs on a general-purpose CPU using plain Python and NumPy. I measured it directly using the professor's transformer code (no PyTorch, just Python arrays and math).

Here is what the measurement showed. For a matrix with 64 rows and 128 numbers per row, the unfused software pipeline takes about 452 microseconds and processes data at only 235 million floating point operations per second. That sounds fast in everyday terms, but it is actually very slow for the hardware -the CPU is spending most of its time waiting for data to come back from memory, not doing math.

The reason is straightforward: the softmax function reads the data three separate times (once to find the maximum, once to compute exponentials, once to divide), and then layer normalization reads it three more times (once for the mean, once for the variance, once to normalize). Six total memory reads for a single normalization step is extremely inefficient.

The fundamental limit is not the speed of the math - it is the number of trips to memory. The arithmetic intensity of this kernel is only 0.27 floating-point operations per byte of data moved, which puts it well below the threshold where the hardware is doing useful work rather than just waiting.

Existing solutions do not help much. General-purpose CPUs cannot skip these memory passes because they process each operation one at a time. GPU libraries like PyTorch can fuse these operations in software, but that requires a GPU and all the overhead that comes with it. There is no open source hardware implementation that runs on the open SKY130 chip process.
```
## Q3: What is new in your approach and why will it succeed?
```
The core idea is simple: instead of reading the data six times, read it once. 

This is possible because of a technique called the Welford online algorithm. Rather than computing the mean and variance in separate passes, Welford's method updates both values one element at a time as each number streams through. The same idea applies to softmax - there is an online version that tracks the running maximum and running sum simultaneously. Together, these two online algorithms let the hardware compute the complete normalized output without ever storing intermediate  results outside the chip.

The effect on arithmetic intensity is significant. The unfused approach has an intensity of 0.27 operations per byte. The fused approach, by cutting memory traffic from six passes to one, raises that to 0.81 operations per byte. When the data is also quantized to 8-bit integers instead of 64-bit floats, it rises further to 6.5 operations per byte. That is a 24-fold improvement in how efficiently the hardware is used, just from these two changes.

On the roofline model, this moves the kernel from the memory-bound region into the compute-bound region. That means the hardware can now run close to its peak throughput rather than sitting idle waiting for data.

In terms of hardware, the design is an eight-stage pipeline. Each stage handles one part of the computation - finding the running maximum, computing the exponential approximation, summing, normalizing, tracking the mean and variance, and applying the final scaling. The pipeline processes one element per clock cycle. At 200 MHz with the target data size, the full computation finishes in about one microsecond, compared to 452 microseconds for the software baseline. 

I think this will work for three reasons. First, the hardware is simple it is a pipeline of small arithmetic units, not a complex array of multipliers, so it is much more likely to synthesize cleanly. Second, the math is well-understood and the error from the 8-entry exponential lookup table is below 0.1%, which is acceptable for normalization in inference. Third, the design is deliberately scoped to one fixed operation with fixed dimensions, which makes timing closure and verification realistic within one term.

```
