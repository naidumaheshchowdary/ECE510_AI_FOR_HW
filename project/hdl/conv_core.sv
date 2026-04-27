// conv_core.sv — Project Compute Core (Stub)
// ECE 510 — Hardware for AI, Spring 2026
// Author: Mahesh Chowdary Naidu
//
// Description:
//   Top-level INT8 convolution compute core. Each cycle, CHANNELS activation-
//   weight pairs are multiplied and accumulated into a 32-bit output register.
//
// Precision : INT8 symmetric per-tensor quantization
// Interface : SPI slave (justified in README — low pin count, sufficient
//             bandwidth for INT8 at target arithmetic intensity ~0.5 MAC/byte)

module conv_core #(
    parameter int DATA_WIDTH  = 8,
    parameter int ACCUM_WIDTH = 32,
    parameter int CHANNELS    = 4
) (
    input  logic                          clk,
    input  logic                          rst,        // active-high synchronous

    input  logic signed [DATA_WIDTH-1:0]  act [CHANNELS-1:0],
    input  logic signed [DATA_WIDTH-1:0]  wgt [CHANNELS-1:0],
    input  logic                          valid_in,

    output logic signed [ACCUM_WIDTH-1:0] accum_out,
    output logic                          valid_out
);

    logic signed [2*DATA_WIDTH-1:0]  products [CHANNELS-1:0];
    logic signed [ACCUM_WIDTH-1:0]   lane_sum;

    // Combinational: multiply each lane and reduce
    always_comb begin
        lane_sum = '0;
        for (int i = 0; i < CHANNELS; i++) begin
            products[i] = $signed(act[i]) * $signed(wgt[i]);
            lane_sum    = lane_sum + ACCUM_WIDTH'(signed'(products[i]));
        end
    end

    // Sequential: accumulate
    always_ff @(posedge clk) begin
        if (rst) begin
            accum_out <= '0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            accum_out <= accum_out + lane_sum;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
