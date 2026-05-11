# CF06 - CMAN: Sneak Paths in a Resistive Crossbar
ECE 410/510, Spring 2026

---

## Setup

The crossbar is 2x2. Row 0 gets 1V applied, col 0 is held at 0V for current
sensing. The four cell resistances are:

- R[0][0] = 1 kΩ  (on)
- R[0][1] = 2 kΩ  (off)
- R[1][0] = 2 kΩ  (off)
- R[1][1] = 1 kΩ  (on)

---

## (a) Ideal Read

For the ideal case, row 1 and col 1 are both grounded. So the only thing
driving current into col 0 is row 0 through R[0][0]. R[1][0] has 0V on both
ends so it contributes nothing.

Just Ohm's law:

    I_col0 = (V_row0 - V_col0) / R[0][0]
           = (1 - 0) / 1000
           = 1 mA

---

## (b) Sneak Path - Finding V_row1 and V_col1

Now row 1 and col 1 are left floating. That means no external current is
forced into or out of those nodes, so by KCL the currents through the
resistors at each floating node have to balance to zero.

I'll call the two unknowns V_row1 and V_col1.

**KCL at V_row1:**

Row 1 connects to col 0 (0V) through R[1][0] = 2k, and to col 1 through
R[1][1] = 1k. Since the node is floating, those two currents must cancel:

    (V_row1 - 0) / 2k  +  (V_row1 - V_col1) / 1k  =  0

Multiply everything by 2k to clear the fractions:

    V_row1 + 2(V_row1 - V_col1) = 0
    3*V_row1 - 2*V_col1 = 0   ... (1)

**KCL at V_col1:**

Col 1 sees current coming in from row 0 through R[0][1] = 2k, and from
row 1 through R[1][1] = 1k. Same deal - floating node, so they sum to zero:

    (1 - V_col1) / 2k  +  (V_row1 - V_col1) / 1k  =  0

Multiply by 2k:

    (1 - V_col1) + 2(V_row1 - V_col1) = 0
    2*V_row1 - 3*V_col1 = -1   ... (2)

**Solving (1) and (2):**

From (1):  V_row1 = (2/3) * V_col1

Plug into (2):

    2*(2/3)*V_col1 - 3*V_col1 = -1
    (4/3 - 9/3)*V_col1 = -1
    (-5/3)*V_col1 = -1
    V_col1 = 3/5 = 0.6 V

Then:  V_row1 = (2/3) * 0.6 = 0.4 V

So the floating nodes settle at **V_row1 = 0.4 V** and **V_col1 = 0.6 V**.

---

## (c) Actual I_col0 with Sneak Path

Col 0 is at 0V. Two resistors dump current into it now - R[0][0] from row 0,
and R[1][0] from the floating row 1 which ended up at 0.4V.

Direct path (what we want):

    I_direct = (1 - 0) / 1k = 1.0 mA

Sneak path (what we don't want):

    I_sneak = (0.4 - 0) / 2k = 0.2 mA

Total current sensed at col 0:

    I_col0 = 1.0 + 0.2 = 1.2 mA

The sneak path adds 0.2 mA that shouldn't be there — that's a 20% error
over the ideal 1 mA.

---

## (d) Why This Corrupts MVM and What It Means for Large Arrays

The whole point of the crossbar is that the current on each output column
equals the dot product of the input voltages with that column's conductances.
Sneak paths break this because a floating row picks up voltage through the
off-state cells connected to the driven row, and that voltage then drives
extra current back into the sense column through a completely different path.
The sensed current no longer reflects only the intended weight — it picks up
contributions from cells that were supposed to be off.

In a large array this gets much worse. Every additional row that's floating
adds another sneak path route, and those paths combine in parallel so the
error current grows with array size. A 128x128 crossbar could have hundreds
of sneak paths all summing into one column at once, completely overwhelming
the real signal. This is why practical crossbar designs use a selector element
(a transistor or diode) at each cell - to block current from flowing unless
that specific row is being actively driven.

---

*Codefest 6, ECE 410/510 Spring 2026*
