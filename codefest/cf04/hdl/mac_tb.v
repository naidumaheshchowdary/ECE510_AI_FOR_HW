// mac_tb.v — Testbench for mac_correct.v
// Sequence: a=3,b=4 for 3 cycles → rst → a=-5,b=2 for 2 cycles

`timescale 1ns/1ps

module mac_tb;
    logic        clk, rst;
    logic signed [7:0]  a, b;
    logic signed [31:0] out;

    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .out(out));

    // 10 ns clock
    initial clk = 0;
    always #5 clk = ~clk;

    task check(input signed [31:0] expected, input string label);
        @(posedge clk); #1;
        if (out !== expected) begin
            $display("FAIL [%s]: got %0d, expected %0d", label, out, expected);
            $finish;
        end else
            $display("PASS [%s]: out = %0d", label, out);
    endtask

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, mac_tb);

        rst = 1; a = 0; b = 0;
        @(posedge clk); #1;
        rst = 0;

        // Cycle 1-3: a=3, b=4 → product=12 each cycle
        a = 3; b = 4;
        check(12,  "cycle1 (3*4=12)");
        check(24,  "cycle2 (12+12=24)");
        check(36,  "cycle3 (24+12=36)");

        // Assert reset
        rst = 1;
        @(posedge clk); #1;
        if (out !== 0) begin
            $display("FAIL [rst]: out = %0d, expected 0", out);
            $finish;
        end else
            $display("PASS [rst]: out = 0");
        rst = 0;

        // Cycle 4-5: a=-5, b=2 → product=-10 each cycle
        a = -5; b = 2;
        check(-10, "cycle4 (-5*2=-10)");
        check(-20, "cycle5 (-10+(-10)=-20)");

        $display("All tests PASSED");
        $finish;
    end
endmodule
