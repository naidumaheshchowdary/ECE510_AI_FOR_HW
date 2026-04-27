# MAC Code Review
## ECE 410/510 — Codefest 4, CLLM

---

## LLM Attribution

| File          | Model              |
|---------------|--------------------|
| `mac_llm_A.v` | Claude Sonnet 4.6  |
| `mac_llm_B.v` | GPT-5.3             |
| `mac_correct.v` | Claude Sonnet 4.6 (corrected) |

---

## Step 2 — Compilation Results

### mac_llm_A.v

```
$ iverilog -g2012 -o test_A mac_llm_A.v
```

**Result: Compiled successfully — no errors.**

> iverilog accepted plain `always @(posedge clk)` without error. The unsigned multiply
> and structural issues are silent functional bugs — not caught at compile time.
> `verilator --lint-only` would flag the non-`always_ff` process in strict SV mode.

Compiled VVP output confirms:
- `.arith/mult 16` — 16-bit multiply with **no sign annotation** (unsigned)
- `%parti/s 1, 15, 5` + `%replicate 16` — manual sign extension of `product[15]`
- `product` computed as a continuous `wire` outside the clocked thread

### mac_llm_B.v

```
$ iverilog -g2012 -o test_B mac_llm_B.v
```

**Result: Compiled successfully — no errors.**

Compiled VVP output confirms:
- `.net/s "a"` and `.net/s "b"` — ports correctly marked signed (`/s`)
- `%pad/s 32` — sign-extends both `a` and `b` to 32 bits before multiply
- `%mul` — 32-bit multiply (correct result, though intermediate sizing differs from explicit 16-bit product)
- `always_ff` inferred correctly as edge-triggered thread

---

## Step 3 — Simulation Results

### Testbench sequence
| Cycle | rst | a  | b | Expected out |
|-------|-----|----|---|--------------|
| 0     | 1   | 0  | 0 | 0            |
| 1     | 0   | 3  | 4 | 12           |
| 2     | 0   | 3  | 4 | 24           |
| 3     | 0   | 3  | 4 | 36           |
| 4     | 1   | —  | — | 0 (reset)    |
| 5     | 0   | -5 | 2 | -10          |
| 6     | 0   | -5 | 2 | -20          |

### mac_llm_A.v simulation

**FAILS on negative input.**

```
$ vvp sim_A
VCD info: dumpfile dump.vcd opened for output.
PASS [cycle1 (3*4=12)]: out = 12
PASS [cycle2 (12+12=24)]: out = 24
PASS [cycle3 (24+12=36)]: out = 36
PASS [rst]: out = 0
FAIL [cycle4 (-5*2=-10)]: got 502, expected -10
mac_tb.v:21: $finish called at 56000 (1ps)
```

Cycles 1–3 pass because unsigned and signed multiplication produce identical results
for positive operands. Cycle 4 fails because `a=-5` is interpreted as unsigned `8'hFB=251`,
giving `251 * 2 = 502` instead of `-10`. Simulation halts at first failure.

### mac_llm_B.v simulation

**PASSES all cycles.**

```
$ vvp sim_B
VCD info: dumpfile dump.vcd opened for output.
PASS [cycle1 (3*4=12)]: out = 12
PASS [cycle2 (12+12=24)]: out = 24
PASS [cycle3 (24+12=36)]: out = 36
PASS [rst]: out = 0
PASS [cycle4 (-5*2=-10)]: out = -10
PASS [cycle5 (-10+(-10)=-20)]: out = -20
All tests PASSED
mac_tb.v:56: $finish called at 66000 (1ps)
```

### mac_correct.v simulation

**PASSES all cycles.**

```
$ vvp sim_correct
VCD info: dumpfile dump.vcd opened for output.
PASS [cycle1 (3*4=12)]: out = 12
PASS [cycle2 (12+12=24)]: out = 24
PASS [cycle3 (24+12=36)]: out = 36
PASS [rst]: out = 0
PASS [cycle4 (-5*2=-10)]: out = -10
PASS [cycle5 (-10+(-10)=-20)]: out = -20
All tests PASSED
mac_tb.v:56: $finish called at 66000 (1ps)
```

---

## Step 4 — Code Review: Issues Found

---

### Issue 1 — Wrong Process Type (mac_llm_A.v)

**Offending lines:**
```systemverilog
always @(posedge clk) begin
```

**Why it is wrong:**
The spec requires `always_ff`. Plain `always` is Verilog-2001 style and is not
synthesizable SystemVerilog. `always_ff` enforces at compile/lint time that the block
infers flip-flops only — no combinational paths, no latches. Tools like Verilator and
Synopsys DC will warn or error on `always` in a `.sv` file.

**Corrected version:**
```systemverilog
always_ff @(posedge clk) begin
```

---

### Issue 2 — Unsigned Multiply on Signed Operands (mac_llm_A.v)

**Offending lines:**
```systemverilog
input  [7:0] a,
input  [7:0] b,
...
wire [15:0] product;
assign product = a * b;
```

**Why it is wrong:**
Ports declared without `signed` are treated as unsigned. `a * b` performs an **unsigned**
16-bit multiply. For negative inputs (e.g. `a=-5 = 8'hFB`), the result is `251*2=502`
instead of the correct `-10`. The downstream sign-extension `{{16{product[15]}}, product}`
then extends the wrong value, silently producing incorrect accumulation for any negative
operand. This bug does **not** appear when both inputs are positive.

**Corrected version:**
```systemverilog
input  logic signed [7:0] a,
input  logic signed [7:0] b,
...
logic signed [15:0] product;
always_ff @(posedge clk) begin
    if (rst)
        out <= 32'sd0;
    else begin
        product = a * b;
        out <= out + 32'(signed'(product));
    end
end
```

---

### Issue 3 — Implicit Product Width in Accumulation (mac_llm_B.v)

**Offending lines:**
```systemverilog
out <= out + (a * b);
```

**Why it is ambiguous:**
Although `a` and `b` are declared `signed`, SystemVerilog sizes the intermediate result
of `a * b` to `max(8, 8) = 8 bits` before the addition context widens it. The compiler
(iverilog) happens to sign-extend correctly here via `%pad/s 32` as seen in the VVP
output, but this relies on implicit context-driven extension. In stricter synthesizers
(Synopsys, Cadence) this can generate a width-mismatch warning or be flagged as
ambiguous. The explicit and portable form uses a 16-bit signed intermediate.

**Corrected version:**
```systemverilog
logic signed [15:0] product;
always_ff @(posedge clk) begin
    if (rst)
        out <= 32'sd0;
    else begin
        product = a * b;                        // explicit 16-bit signed product
        out     <= out + 32'(signed'(product)); // unambiguous sign-extension
    end
end
```

---

## Step 5 — mac_correct.v

```systemverilog
// mac_correct.v — Claude Sonnet 4.6

module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [31:0] out
);

    logic signed [15:0] product;

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else begin
            product = a * b;
            out     <= out + 32'(signed'(product));
        end
    end

endmodule
```

**Compliance checklist:**

| Requirement                          | Status |
|--------------------------------------|--------|
| Module name `mac`                    | ✅     |
| `clk` 1-bit input                    | ✅     |
| `rst` active-high synchronous reset  | ✅     |
| `a` 8-bit signed input               | ✅     |
| `b` 8-bit signed input               | ✅     |
| `out` 32-bit signed accumulator      | ✅     |
| `always_ff` used                     | ✅     |
| No `initial` blocks                  | ✅     |
| No `$display`                        | ✅     |
| No delays (`#`)                      | ✅     |
| Correct sign extension 16→32 bits    | ✅     |
| Synthesizable SystemVerilog only     | ✅     |
| Compiles cleanly                     | ✅     |
| Passes testbench                     | ✅     |
