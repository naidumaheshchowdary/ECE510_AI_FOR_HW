// mac_correct.v
// Correct synthesizable SystemVerilog MAC unit
// Model: Claude Sonnet 4.6

module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [31:0] out
);

    logic signed [15:0] product;
    logic signed [31:0] product_ext;

    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            product     = a * b;                          // 16-bit signed multiply
            product_ext = {{16{product[15]}}, product};   // explicit sign-extend to 32-bit
            out         <= out + product_ext;
        end
    end

endmodule
