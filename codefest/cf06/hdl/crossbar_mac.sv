// =============================================================
// crossbar_mac.sv — 4×4 Binary-Weight Crossbar MAC Unit
// ECE 410/510 Spring 2026 — Codefest 6 (CLLM)
//
// out[j] = Σ_i weight[i][j] × in[i],  weight ∈ {+1,−1}
// Inputs : 4 × 8-bit signed
// Outputs: 4 × 16-bit signed (packed into 64-bit bus)
// Weights: loaded row-by-row via w_row / w_data
// =============================================================

module crossbar_mac (
    input  logic        clk,
    input  logic        rst,
    input  logic        load_weights,
    input  logic [1:0]  w_row,            // 0-3: which weight row to load
    input  logic [3:0]  w_data,           // bit[j]=1 → +1, bit[j]=0 → -1
    // Input vector (8-bit signed each)
    input  logic signed [7:0]  in0, in1, in2, in3,
    // Output accumulators (16-bit signed each)
    output logic signed [15:0] out0, out1, out2, out3
);

    // 4×4 weight register; w[i][j] stored as w[i] bit j
    logic [3:0] w [3:0];

    // Sign-extend inputs to 16 bits
    wire signed [15:0] e0 = {{8{in0[7]}}, in0};
    wire signed [15:0] e1 = {{8{in1[7]}}, in1};
    wire signed [15:0] e2 = {{8{in2[7]}}, in2};
    wire signed [15:0] e3 = {{8{in3[7]}}, in3};

    // Cell contributions  cIJ = ±eI  based on weight[I][J]
    wire signed [15:0] c00=w[0][0]?e0:-e0; wire signed [15:0] c10=w[1][0]?e1:-e1;
    wire signed [15:0] c20=w[2][0]?e2:-e2; wire signed [15:0] c30=w[3][0]?e3:-e3;

    wire signed [15:0] c01=w[0][1]?e0:-e0; wire signed [15:0] c11=w[1][1]?e1:-e1;
    wire signed [15:0] c21=w[2][1]?e2:-e2; wire signed [15:0] c31=w[3][1]?e3:-e3;

    wire signed [15:0] c02=w[0][2]?e0:-e0; wire signed [15:0] c12=w[1][2]?e1:-e1;
    wire signed [15:0] c22=w[2][2]?e2:-e2; wire signed [15:0] c32=w[3][2]?e3:-e3;

    wire signed [15:0] c03=w[0][3]?e0:-e0; wire signed [15:0] c13=w[1][3]?e1:-e1;
    wire signed [15:0] c23=w[2][3]?e2:-e2; wire signed [15:0] c33=w[3][3]?e3:-e3;

    // Column sums (combinational)
    wire signed [15:0] acc0 = c00+c10+c20+c30;
    wire signed [15:0] acc1 = c01+c11+c21+c31;
    wire signed [15:0] acc2 = c02+c12+c22+c32;
    wire signed [15:0] acc3 = c03+c13+c23+c33;

    // Sequential: reset / weight load / output register
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            w[0]<=4'b0; w[1]<=4'b0; w[2]<=4'b0; w[3]<=4'b0;
            out0<=0; out1<=0; out2<=0; out3<=0;
        end
        else if (load_weights) begin
            w[w_row] <= w_data;
        end
        else begin
            out0 <= acc0;
            out1 <= acc1;
            out2 <= acc2;
            out3 <= acc3;
        end
    end

endmodule