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

**max(|W|) = 2.31**  (element W[2][3])

**S = max(|W|) / 127 = 2.31 / 127 = 0.0182**

---

## (b) Quantized INT8 Matrix W_q

Formula: `W_q = round(W / S)`, clamped to [−128, 127]

Pre-round values (W / S):

```
[[ 46.70,  -65.93,  18.68,  115.38 ]
 [ -3.85,   50.00, -103.30,   6.59 ]
 [ 85.16,    1.65,  -24.18, -126.92 ]
 [ -9.89,   56.59,   42.31,  30.22 ]]
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

Formula: `W_deq = W_q × S`

```
[[  0.8554,  -1.2012,   0.3458,   2.0930 ]
 [ -0.0728,   0.9100,  -1.8746,   0.1274 ]
 [  1.5470,   0.0364,  -0.4368,  -2.3114 ]
 [ -0.1820,   1.0374,   0.7644,   0.5460 ]]
```

---

## (d) Error Analysis

Per-element absolute error |W − W_deq|:

```
[[ 0.0054,  0.0012,  0.0058,  0.0070 ]
 [ 0.0028,  0.0000,  0.0054,  0.0074 ]
 [ 0.0030,  0.0064,  0.0032,  0.0014 ]
 [ 0.0020,  0.0074,  0.0056,  0.0040 ]]
```

**Largest error: 0.0074** — occurs at two elements:
- W[1][3] = 0.12  → W_deq = 0.1274
- W[3][1] = 1.03  → W_deq = 1.0374

**Sum of errors = 0.0536**

**MAE = 0.0536 / 16 = 0.00335**

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

Note: elements W[0][3]=2.10, W[2][0]=1.55, W[1][2]=−1.88, W[2][3]=−2.31 are **clamped**
(their true quantized values would exceed ±128).

**W_deq_bad = W_q_bad × 0.01:**

```
[[ 0.85,  -1.20,   0.34,   1.27 ]
 [-0.07,   0.91,  -1.28,   0.12 ]
 [ 1.27,   0.03,  -0.44,  -1.28 ]
 [-0.18,   1.03,   0.77,   0.55 ]]
```

**Per-element errors |W − W_deq_bad|:**

```
[[ 0.00,  0.00,  0.00,  0.83 ]
 [ 0.00,  0.00,  0.60,  0.00 ]
 [ 0.28,  0.00,  0.00,  1.03 ]
 [ 0.00,  0.00,  0.00,  0.00 ]]
```

**MAE_bad = 0.1713**  (vs MAE = 0.00335 with correct S)

**Explanation:** When S is too small, large weight values exceed the INT8 representable
range and are hard-clamped to ±127/−128, causing severe saturation clipping errors that
cannot be recovered during dequantization.
