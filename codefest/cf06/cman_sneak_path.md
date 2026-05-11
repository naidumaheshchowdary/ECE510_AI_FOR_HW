# CMAN — Sneak Paths in a Resistive Crossbar
**ECE 410/510 — Spring 2026 — Codefest 6**

---

## Circuit Setup

2×2 resistive crossbar with fixed cell resistances:

| Cell     | Resistance | Weight |
|----------|-----------|--------|
| R[0][0]  | 1 kΩ      | ON     |
| R[0][1]  | 2 kΩ      | OFF    |
| R[1][0]  | 2 kΩ      | OFF    |
| R[1][1]  | 1 kΩ      | ON     |

- Rows carry input voltages; columns carry output currents.
- V_row0 = 1 V (driven), V_col0 = 0 V (virtual ground / sense)

---

## (a) Ideal Read — I_col0

**Conditions:** V_row0 = 1 V, V_col0 = 0 V, V_row1 = 0 V, V_col1 = 0 V (all grounded except row 0)

With row1 and col1 grounded, only R[0][0] connects V_row0 to col0.
R[1][0] connects row1 (0 V) to col0 (0 V) - zero voltage drop, zero current.

**Calculation:**

```
I_col0 = (V_row0 - V_col0) / R[0][0]
       = (1 V - 0 V) / 1000 Ω
       = 1 mA
```

**Ideal I_col0 = 1 mA**

---

## (b) KCL Solution - Floating Node Voltages V_row1 and V_col1

**Conditions:** V_row0 = 1 V, V_col0 = 0 V, V_row1 = floating, V_col1 = floating

Since row1 and col1 are undriven (floating), no current can enter or leave
those nodes externally. By KCL, the sum of currents at each floating node = 0.

### KCL at V_row1:

Currents leaving V_row1 through R[1][0] and R[1][1] must sum to zero:

```
(V_row1 - V_col0) / R[1][0]  +  (V_row1 - V_col1) / R[1][1]  =  0

(V_row1 - 0) / 2k  +  (V_row1 - V_col1) / 1k  =  0
```

Multiply through by 2k:

```
V_row1  +  2(V_row1 - V_col1)  =  0
3·V_row1  -  2·V_col1  =  0   ... (Equation 1)
```

### KCL at V_col1:

Currents flowing into V_col1 from V_row0 (through R[0][1]) and from V_row1
(through R[1][1]) must sum to zero:

```
(V_row0 - V_col1) / R[0][1]  +  (V_row1 - V_col1) / R[1][1]  =  0

(1 - V_col1) / 2k  +  (V_row1 - V_col1) / 1k  =  0
```

Multiply through by 2k:

```
(1 - V_col1)  +  2(V_row1 - V_col1)  =  0
2·V_row1  -  3·V_col1  =  -1          ... (Equation 2)
```

### Solving the system:

From Equation 1:
```
V_row1 = (2/3) · V_col1
```

Substitute into Equation 2:
```
2 · (2/3) · V_col1  -  3 · V_col1  =  -1
(4/3) · V_col1  -  3 · V_col1      =  -1
(4/3 - 9/3) · V_col1               =  -1
(-5/3) · V_col1                     =  -1

V_col1 = 3/5 = 0.6 V
V_row1 = (2/3) × 0.6 = 0.4 V
```

**V_col1 = 0.6 V**
**V_row1 = 0.4 V**

---

## (c) Actual I_col0 with Sneak Path Current Itemized

Col0 is held at 0 V. Two resistors connect to col0:

### Direct path (intended):
```
I_direct = (V_row0 - V_col0) / R[0][0]
         = (1 V - 0 V) / 1 kΩ
         = 1.0 mA
```

### Sneak path (unintended):
```
I_sneak  = (V_row1 - V_col0) / R[1][0]
         = (0.4 V - 0 V) / 2 kΩ
         = 0.2 mA
```

### Total actual current:
```
I_col0_actual = I_direct + I_sneak
              = 1.0 mA + 0.2 mA
              = 1.2 mA
```

| Path        | Current  | Description                        |
|-------------|----------|------------------------------------|
| Direct      | 1.0 mA   | V_row0 → R[0][0] → col0  (correct) |
| Sneak       | 0.2 mA   | V_row1 → R[1][0] → col0  (error)   |
| **Total**   | **1.2 mA** | **20% error over ideal**         |

---

## (d) How Sneak Paths Corrupt MVM Results

In a resistive crossbar MVM, the output current on each column is supposed to
represent the dot product of the input voltage vector with the column's weight
vector. Sneak paths create unintended current loops through floating rows and
columns — current from an active row leaks through off-state cells into floating
nodes and then back into the sense column through a different resistor, adding
spurious current that was never part of the intended computation. In large
crossbar arrays this effect worsens significantly because every additional row
and column adds more parallel sneak-path routes, causing output currents to
deviate further from their ideal values and making it impossible to distinguish
correct MVM results from noise without active mitigation such as selector devices
(e.g. transistors or diodes) at each cell.

---

*ECE 410/510 Spring 2026 — Codefest 6 — CMAN (No AI)*

