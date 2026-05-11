// =============================================================
// crossbar_mac.sv
// 4x4 Binary-Weight Crossbar MAC Unit
// ECE 410/510 Spring 2026 — Codefest 6 (CLLM task)
//
// Description:
//   Computes out[j] = Σ_i weight[i][j] × in[i]
//   Weights are binary: reg=1 → +1, reg=0 → -1
//   Inputs: 4 × 8-bit signed
//   Outputs: 4 × 16-bit signed accumulators
// =============================================================

module crossbar_mac (
    input  logic        clk,
    input  logic        rst,
    input  logic        load_weights,           // pulse high to latch new weights
    input  logic [3:0]  weight_in [3:0],        // weight_in[i][j]: 1=+1, 0=-1
    input  logic signed [7:0]  in_data [3:0],   // 4 x 8-bit signed inputs
    output logic signed [15:0] out [3:0]         // 4 x 16-bit signed outputs
);

    // -------------------------------------------------------
    // Internal weight register array: weight_reg[i][j]
    // weight_reg[i][j] = 1 → +1 weight
    // weight_reg[i][j] = 0 → -1 weight
    // -------------------------------------------------------
    logic [3:0] weight_reg [3:0];   // weight_reg[i] = 4-bit row for input i

    // -------------------------------------------------------
    // Sequential logic: reset, weight load, MAC computation
    // -------------------------------------------------------
    integer i, j;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // Clear all outputs and weights
            for (i = 0; i < 4; i++) begin
                out[i]        <= 16'sd0;
                weight_reg[i] <= 4'b0000;
            end
        end
        else if (load_weights) begin
            // Latch incoming weight matrix
            for (i = 0; i < 4; i++) begin
                weight_reg[i] <= weight_in[i];
            end
        end
        else begin
            // Compute MAC: out[j] = Σ_i (weight[i][j]==1 ? +in[i] : -in[i])
            for (j = 0; j < 4; j++) begin
                out[j] <= 16'sd0;
                for (i = 0; i < 4; i++) begin
                    if (weight_reg[i][j] == 1'b1)
                        out[j] <= out[j] + 16'(signed'(in_data[i]));
                    else
                        out[j] <= out[j] - 16'(signed'(in_data[i]));
                end
            end
        end
    end

endmodule
