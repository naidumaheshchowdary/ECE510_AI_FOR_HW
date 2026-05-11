// =============================================================
// crossbar_tb.sv — Testbench for crossbar_mac
// ECE 410/510 Spring 2026 — Codefest 6 (CLLM)
//
// Weight matrix:
//   W = [[ 1,-1, 1,-1],   row i=0 → w_data=4'b0101
//        [ 1, 1,-1,-1],   row i=1 → w_data=4'b0011
//        [-1, 1, 1,-1],   row i=2 → w_data=4'b0110
//        [-1,-1,-1, 1]]   row i=3 → w_data=4'b1000
//
// Input: [10, 20, 30, 40]
// Expected:
//   out0 = +10+20-30-40 = -40
//   out1 = -10+20+30-40 =   0
//   out2 = +10-20+30-40 = -20
//   out3 = -10-20-30+40 = -20
// =============================================================

`timescale 1ns/1ps
module crossbar_tb;

    logic        clk, rst, load_weights;
    logic [1:0]  w_row;
    logic [3:0]  w_data;
    logic signed [7:0]  in0,in1,in2,in3;
    logic signed [15:0] out0,out1,out2,out3;

    crossbar_mac dut(
        .clk(clk),.rst(rst),
        .load_weights(load_weights),
        .w_row(w_row),.w_data(w_data),
        .in0(in0),.in1(in1),.in2(in2),.in3(in3),
        .out0(out0),.out1(out1),.out2(out2),.out3(out3)
    );

    initial clk=0;
    always #5 clk=~clk;

    task check(input string s,
               input logic signed [15:0] got,
               input shortint exp);
        if(got===exp) $display("  PASS  %s = %0d (expected %0d)",s,got,exp);
        else          $display("  FAIL  %s = %0d (expected %0d) ***",s,got,exp);
    endtask

    initial begin
        $dumpfile("crossbar_tb.vcd"); $dumpvars(0,crossbar_tb);
        $display("=================================================");
        $display(" crossbar_mac Testbench - ECE 410/510 CF06");
        $display("=================================================");

        // Reset
        rst=1; load_weights=0; w_row=0; w_data=0;
        in0=0; in1=0; in2=0; in3=0;
        @(posedge clk); #1; rst=0;

        // Load weights row by row
        $display("\n[1] Loading weight matrix...");
        load_weights=1;
        w_row=0; w_data=4'b0101; @(posedge clk); #1;  // row0:[ 1,-1, 1,-1]
        w_row=1; w_data=4'b0011; @(posedge clk); #1;  // row1:[ 1, 1,-1,-1]
        w_row=2; w_data=4'b0110; @(posedge clk); #1;  // row2:[-1, 1, 1,-1]
        w_row=3; w_data=4'b1000; @(posedge clk); #1;  // row3:[-1,-1,-1, 1]
        load_weights=0;
        $display("    w[0]=0101 w[1]=0011 w[2]=0110 w[3]=1000");

        // Apply inputs
        $display("\n[2] Applying inputs [10,20,30,40]...");
        in0=8'sd10; in1=8'sd20; in2=8'sd30; in3=8'sd40;
        @(posedge clk); #1;
        @(posedge clk); #1;

        // Check
        $display("\n[3] Results:");
        $display("-------------------------------------------------");
        check("out0", out0, -40);
        check("out1", out1,   0);
        check("out2", out2, -20);
        check("out3", out3, -20);
        $display("-------------------------------------------------");
        $display("[Raw] out0=%0d out1=%0d out2=%0d out3=%0d",
                  out0,out1,out2,out3);
        $display("\n=================================================");
        $display(" Simulation complete.");
        $display("=================================================");
        $finish;
    end
endmodule