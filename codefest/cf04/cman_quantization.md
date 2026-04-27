# CMAN — Manual INT8 Symmetric Quantization
## ECE 410/510 — Codefest 4

---

## (a) Scale Factor

Given weight matrix W:

```
W = [  0.85,  -1.20,   0.34,   2.10 ]
    [ -0.07,   0.91,  -1.88,   0.12 ]
    [  1.55,   0.03,  -0.44,  -2.31 ]
    [ -0.18,   1.03,   0.77,   0.55 ]
```

Formula: `S = max(|W|) / 127`

Absolute values scanned across all 16 elements:

```
|W| = [[ 0.85,  1.20,  0.34,  2.10 ]
       [ 0.07,  0.91,  1.88,  0.12 ]
       [ 1.55,  0.03,  0.44,  2.31 ]  ← maximum here
       [ 0.18,  1.03,  0.77,  0.55 ]]
```

**max(|W|) = 2.31**  (element W[2][3])

**S = 2.31 / 127 = 0.018189**

---

## (b) Quantized INT8 Matrix W_q

Formula: `W_q = round(W / S)`, clamped to [−128, 127]

Pre-round values (W / S):

```
[[ 46.7316,  -65.9740,   18.6926,  115.4545 ]
 [ -3.8485,   50.0303, -103.3593,    6.5974 ]
 [ 85.2165,    1.6494,  -24.1905, -127.0000 ]
 [ -9.8961,   56.6277,   42.3333,   30.2381 ]]
```

**W_q (rounded and clamped):**

```
[[  47,  -66,   19,  115 ]
 [  -4,   50, -103,    7 ]
 [  85,    2,  -24, -127 ]
 [ -10,   57,   42,   30 ]]
```

No values required clamping (all within [−128, 127]).

---

## (c) Dequantized FP32 Matrix W_deq

Formula: `W_deq = W_q × S`  where S = 0.018189

```
[[  0.8549,  -1.2005,   0.3456,   2.0917 ]
 [ -0.0728,   0.9094,  -1.8735,   0.1273 ]
 [  1.5461,   0.0364,  -0.4365,  -2.3100 ]
 [ -0.1819,   1.0368,   0.7639,   0.5457 ]]
```

---

## (d) Error Analysis

Per-element absolute error `|W − W_deq|`:

```
[[ 0.0049,  0.0005,  0.0056,  0.0083 ]
 [ 0.0028,  0.0006,  0.0065,  0.0073 ]
 [ 0.0039,  0.0064,  0.0035,  0.0000 ]
 [ 0.0019,  0.0068,  0.0061,  0.0043 ]]
```

**Largest error: 0.0083** — element W[0][3] = 2.10 → W_deq = 2.0917

**Sum of errors = 0.0692**

**MAE = 0.0692 / 16 = 0.004326**

---

## (e) Bad Scale Experiment (S_bad = 0.01)

Using S_bad = 0.01 (too small):

**W_q_bad = round(W / 0.01), clamped to [−128, 127]:**

```
[[  85, -120,   34,  127 ]
 [  -7,   91, -128,   12 ]
 [ 127,    3,  -44, -128 ]
 [ -18,  103,   77,   55 ]]
```

Clamped elements (true quantized value exceeded ±128):

| Element | W value | Raw W/S_bad | Clamped to |
|---------|---------|-------------|------------|
| W[0][3] |  2.10   |     210     |    127     |
| W[1][2] | -1.88   |    -188     |   -128     |
| W[2][0] |  1.55   |     155     |    127     |
| W[2][3] | -2.31   |    -231     |   -128     |

**W_deq_bad = W_q_bad × 0.01:**

```
[[  0.85,  -1.20,   0.34,   1.27 ]
 [ -0.07,   0.91,  -1.28,   0.12 ]
 [  1.27,   0.03,  -0.44,  -1.28 ]
 [ -0.18,   1.03,   0.77,   0.55 ]]
```

**Per-element errors |W − W_deq_bad|:**

```
[[ 0.00,  0.00,  0.00,  0.83 ]
 [ 0.00,  0.00,  0.60,  0.00 ]
 [ 0.28,  0.00,  0.00,  1.03 ]
 [ 0.00,  0.00,  0.00,  0.00 ]]
```

**MAE_bad = 0.1713**  (vs MAE = 0.004326 with correct S — roughly **40× worse**)

**Explanation:** When S is too small, large weight values exceed the INT8
representable range [−128, 127] and are hard-clamped, causing large
irrecoverable saturation errors that cannot be corrected during dequantization.
