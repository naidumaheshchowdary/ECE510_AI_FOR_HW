# CMAN -- Systolic Array Trace (Weight-Stationary)
## ECE 410/510 HW4AI | Codefest 5 | Spring 2026

Given:  
A = [[1, 2], [3, 4]]  
B = [[5, 6], [7, 8]]  
Expected output C = [[19, 22], [43, 50]]

---

## (a) PE Diagram

A 2×2 weight-stationary systolic array has 4 processing elements arranged
in 2 rows and 2 columns. Before the computation starts, the B matrix
values are loaded into the PEs and stay there the whole time - that is
what "weight-stationary" means. The A values then stream in from the top
and partial sums pass downward from one row to the next.

Each PE does one thing: it takes the input coming in from above, multiplies
it by its stored weight, adds that to the partial sum arriving from the PE
above it, and passes the new partial sum down.

```
          col 0                col 1
      ┌───────────┐        ┌───────────┐
      │  PE[0][0] │        │  PE[0][1] │
      │  weight=5 │        │  weight=6 │
      │ (B[0][0]) │        │ (B[0][1]) │
      └─────┬─────┘        └─────┬─────┘
            │ partial sum         │ partial sum
            ▼                    ▼
      ┌───────────┐        ┌───────────┐
      │  PE[1][0] │        │  PE[1][1] │
      │  weight=7 │        │  weight=8 │
      │ (B[1][0]) │        │ (B[1][1]) │
      └─────┬─────┘        └─────┬─────┘
            ▼                    ▼
          C[i][0]             C[i][1]
```

Preloaded weights:

| PE       | Holds weight | Value |
|----------|-------------|-------|
| PE[0][0] | B[0][0]     | 5     |
| PE[0][1] | B[0][1]     | 6     |
| PE[1][0] | B[1][0]     | 7     |
| PE[1][1] | B[1][1]     | 8     |

---

## (b) Cycle-by-Cycle Trace

The two rows of the systolic array correspond to the two terms in the
dot product (k=0 and k=1). A[i][0] feeds into PE row 0 and A[i][1] feeds
into PE row 1. Because the two terms need to meet at the right time,
row 1 inputs are delayed by one cycle relative to row 0. This staggering
is the standard way a systolic array keeps everything aligned.

Input schedule:

- Cycle 1: row 0 gets A[i][0] for i=0, so input = 1. Row 1 gets nothing yet.
- Cycle 2: row 0 gets A[i][0] for i=1, so input = 3. Row 1 gets A[i][1] for i=0, so input = 2.
- Cycle 3: row 0 is done. Row 1 gets A[i][1] for i=1, so input = 4.
- Cycle 4: both rows idle, computation complete.

Each PE rule: `new_psum = psum_arriving_from_above + (input × my_weight)`

| Cycle | Row 0 input | Row 1 input | PE[0][0] | PE[0][1] | PE[1][0] | PE[1][1] | Output |
|-------|------------|------------|---------|---------|---------|---------|--------|
| 1 | 1 (A[0][0]) | 0 | 0+1×5 = **5** | 0+1×6 = **6** | 0+0×7+0 = **0** | 0+0×8+0 = **0** | — |
| 2 | 3 (A[1][0]) | 2 (A[0][1]) | 0+3×5 = **15** | 0+3×6 = **18** | 5+2×7 = **19** | 6+2×8 = **22** | C[0][0]=19, C[0][1]=22 |
| 3 | 0 | 4 (A[1][1]) | 0 | 0 | 15+4×7 = **43** | 18+4×8 = **50** | C[1][0]=43, C[1][1]=50 |
| 4 | - | - | 0 | 0 | 0 | 0 | done |

Walking through the key steps to make sure the numbers make sense:

Cycle 1: A[0][0]=1 enters row 0. PE[0][0] computes 1×5=5 and PE[0][1]
computes 1×6=6. These partial sums sit waiting to pass down. Row 1 has
no input yet so nothing happens there.

Cycle 2: Two things happen at once. Row 0 gets A[1][0]=3, so PE[0][0]
starts a fresh accumulation for the second output row: 3×5=15. At the
same time row 1 finally gets A[0][1]=2. PE[1][0] receives the partial
sum of 5 from PE[0][0] last cycle and adds 2×7=14, giving 5+14=19.
That is C[0][0] and it matches. Similarly PE[1][1] gets 6+2×8=22=C[0][1].

Cycle 3: Row 0 is idle. Row 1 gets A[1][1]=4. PE[1][0] receives 15
from PE[0][0] and computes 15+4×7=15+28=43=C[1][0]. PE[1][1] gets
18+4×8=18+32=50=C[1][1]. Both match the expected output.

Final check:
- C[0][0] = 1×5 + 2×7 = 5+14 = 19 ✓
- C[0][1] = 1×6 + 2×8 = 6+16 = 22 ✓
- C[1][0] = 3×5 + 4×7 = 15+28 = 43 ✓
- C[1][1] = 3×6 + 4×8 = 18+32 = 50 ✓

---

## (c) Counts

### Total MAC operations

Each of the 4 output elements needs 2 multiply-accumulate operations,
one for k=0 and one for k=1:

    Total MACs = 2 × 2 × 2 = 8 MACs

### Input value reuse

Every time an A value enters the array it feeds into both columns at
once - col 0 and col 1 use the same input in the same cycle. So each
A element is used exactly twice.

    Reuse factor = 2  (each A[i][k] feeds PE[k][0] and PE[k][1])

For example A[0][0]=1 is multiplied by weight 5 in PE[0][0] and by
weight 6 in PE[0][1] in the same cycle. Same story for all 4 A values.

### Off-chip memory accesses

| Matrix | Access type | Count | Notes |
|--------|------------|-------|-------|
| A | Read | 4 | Each of the 4 elements loaded once |
| B | Read | 4 | All 4 weights loaded once at startup, never reloaded |
| C | Write | 4 | Each output element written once when it exits |

    Total = 4 + 4 + 4 = 12 off-chip accesses

The main advantage of weight-stationary is that B is only read from
DRAM once. In a naive triple loop B would be reread repeatedly, which
for large matrices is a big memory cost.

---

## (d) Output-Stationary

In output-stationary dataflow, the partial sums for each output element
C[i][j] stay fixed inside their PE for the entire computation, while
the A and B values are the ones that stream in from outside each cycle.
