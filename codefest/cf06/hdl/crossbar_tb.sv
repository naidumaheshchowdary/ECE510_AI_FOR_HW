// =============================================================
// crossbar_tb.sv
// Testbench for crossbar_mac module
// ECE 410/510 Spring 2026 — Codefest 6 (CLLM task)
//
// Weight matrix loaded:
//   W = [[ 1,-1, 1,-1],   ← row 0 (input 0)
//        [ 1, 1,-1,-1],   ← row 1 (input 1)
//        [-1, 1, 1,-1],   ← row 2 (input 2)
//        [-1,-1,-1, 1]]   ← row 3 (input 3)
//
// Input vector: in = [10, 20, 30, 40]
//
// Expected outputs (hand-calculated):
//   out[0] = (+1×10) + (+1×20) + (-1×30) + (-1×40) = -40
//   out[1] = (-1×10) + (+1×20) + (+1×30) + (-1×40) =   0
//   out[2] = (+1×10) + (-1×20) + (+1×30) + (-1×40) = -20
//   out[3] = (-1×10) + (-1×20) + (-1×30) + (+1×40) = -20
// =============================================================

`timescale 1ns/1ps

module crossbar_tb;

    // -------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------
    logic        clk;
    logic        rst;
    logic        load_weights;
    logic [3:0]  weight_in [3:0];
    logic signed [7:0]  in_data [3:0];
    logic signed [15:0] out [3:0];

    // -------------------------------------------------------
    // Instantiate DUT
    // -------------------------------------------------------
    crossbar_mac dut (
        .clk          (clk),
        .rst          (rst),
        .load_weights (load_weights),
        .weight_in    (weight_in),
        .in_data      (in_data),
        .out          (out)
    );

    // -------------------------------------------------------
    // Clock generation: 10 ns period (100 MHz)
    // -------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // -------------------------------------------------------
    // Helper task: check output against expected value
    // -------------------------------------------------------
    task check_output(input int j, input shortint expected);
        if (out[j] === expected)
            $display("  PASS  out[%0d] = %0d (expected %0d)", j, out[j], expected);
        else
            $display("  FAIL  out[%0d] = %0d (expected %0d) *** MISMATCH ***", j, out[j], expected);
    endtask

    // -------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------
    initial begin
        $display("=================================================");
        $display(" crossbar_mac Testbench — ECE 410/510 CF06");
        $display("=================================================");

        // --- 1. Reset ---
        rst          = 1;
        load_weights = 0;
        in_data[0]   = 8'sd0;
        in_data[1]   = 8'sd0;
        in_data[2]   = 8'sd0;
        in_data[3]   = 8'sd0;
        // weight_in: encode +1 as 1, -1 as 0
        // W row 0: [ 1,-1, 1,-1] → bits [j=3,j=2,j=1,j=0] = [0,1,0,1] = 4'b0101
        // W row 1: [ 1, 1,-1,-1] → [0,0,1,1]               = 4'b0011  (j3=−1→0,j2=−1→0,j1=+1→1,j0=+1→1)
        // W row 2: [-1, 1, 1,-1] → [0,1,1,0]               = 4'b0110
        // W row 3: [-1,-1,-1, 1] → [1,0,0,0]               = 4'b1000
        //
        // Encoding reminder: weight_in[i][j], bit j of weight_in[i]
        //   weight_in[0] = {W[0][3], W[0][2], W[0][1], W[0][0]}
        //                = {    -1,      +1,      -1,      +1 }
        //                = {     0,       1,       0,       1 } = 4'b0101
        weight_in[0] = 4'b0101;   // row 0: W=[+1,-1,+1,-1]
        weight_in[1] = 4'b0011;   // row 1: W=[+1,+1,-1,-1]
        weight_in[2] = 4'b0110;   // row 2: W=[-1,+1,+1,-1]
        weight_in[3] = 4'b1000;   // row 3: W=[-1,-1,-1,+1]

        @(posedge clk); #1;
        rst = 0;

        // --- 2. Load weights ---
        $display("\n[Cycle 1] Loading weight matrix...");
        load_weights = 1;
        @(posedge clk); #1;
        load_weights = 0;

        // --- 3. Apply input vector [10, 20, 30, 40] ---
        $display("[Cycle 2] Applying inputs: in=[10, 20, 30, 40]");
        in_data[0] = 8'sd10;
        in_data[1] = 8'sd20;
        in_data[2] = 8'sd30;
        in_data[3] = 8'sd40;
        @(posedge clk); #1;

        // --- 4. Wait one more cycle for output to settle ---
        @(posedge clk); #1;

        // --- 5. Check results ---
        $display("\n[Results] Checking outputs:");
        $display("-------------------------------------------------");
        check_output(0, -40);
        check_output(1,   0);
        check_output(2, -20);
        check_output(3, -20);
        $display("-------------------------------------------------");

        // --- 6. Display raw output values ---
        $display("\n[Raw] out[0]=%0d  out[1]=%0d  out[2]=%0d  out[3]=%0d",
                  out[0], out[1], out[2], out[3]);

        $display("\n=================================================");
        $display(" Simulation complete.");
        $display("=================================================");
        $finish;
    end

    // -------------------------------------------------------
    // Waveform dump (for GTKWave or similar)
    // -------------------------------------------------------
    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);
    end

endmodule
