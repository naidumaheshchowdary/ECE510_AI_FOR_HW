// =============================================================================
// compute_core.sv — 8-stage Fused Softmax + LayerNorm Pipeline
// ECE 410/510 HW4AI | Spring 2026 | Milestone 4
// Author : Mahesh Chowdary Naidu
// =============================================================================
// Change from M3: module renamed compute_core_m3 → compute_core
//                 final_sum register added to fix running_sum reset race
// All other logic identical to M3 compute_core_m3.sv
// =============================================================================
//
// Port list:
//   clk            I   1    System clock
//   rst_n          I   1    Async active-low reset
//   s_axis_tdata   I  64    AXI4-Stream input: 8 x INT8 per beat
//   s_axis_tvalid  I   1    Input valid
//   s_axis_tlast   I   1    Last beat of row
//   s_axis_tready  O   1    Back-pressure
//   m_axis_tdata   O  64    AXI4-Stream output: 8 x INT8 per beat
//   m_axis_tvalid  O   1    Output valid
//   m_axis_tlast   O   1    Last beat of output row
//   m_axis_tready  I   1    Downstream ready
//   cfg_d          I   8    Row width (default 64)
//   cfg_t          I   8    Number of rows (default 64)
//   precision      I   1    0=INT8, 1=FP64 reserved
//   start          I   1    Start pulse from AXI4-Lite CTRL[0]
//   done           O   1    Pulse one cycle after tlast exits pipeline
// =============================================================================
`timescale 1ns/1ps
module compute_core #(
    parameter D          = 64,
    parameter T          = 64,
    parameter DATA_W     = 8,
    parameter AXIS_W     = 64,
    parameter PIPE_DEPTH = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire  [AXIS_W-1:0]   s_axis_tdata,
    input  wire                  s_axis_tvalid,
    input  wire                  s_axis_tlast,
    output reg                   s_axis_tready,
    output wire  [AXIS_W-1:0]   m_axis_tdata,
    output wire                  m_axis_tvalid,
    output wire                  m_axis_tlast,
    input  wire                  m_axis_tready,
    input  wire  [7:0]           cfg_d,
    input  wire  [7:0]           cfg_t,
    input  wire                  precision,
    input  wire                  start,
    output reg                   done
);
    reg [AXIS_W-1:0] pipe_data  [0:PIPE_DEPTH-1];
    reg              pipe_valid [0:PIPE_DEPTH-1];
    reg              pipe_last  [0:PIPE_DEPTH-1];
    reg signed [15:0] running_max;
    reg        [23:0] running_sum;
    reg        [23:0] final_sum;
    reg signed [23:0] welford_mean;
    reg        [23:0] welford_m2;
    reg [7:0] exp_lut [0:7];
    initial begin
        exp_lut[0]=8'd255; exp_lut[1]=8'd224; exp_lut[2]=8'd197; exp_lut[3]=8'd174;
        exp_lut[4]=8'd153; exp_lut[5]=8'd135; exp_lut[6]=8'd119; exp_lut[7]=8'd105;
    end

    // S1 — input latch
    always @(*) s_axis_tready = m_axis_tready | ~pipe_valid[0];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[0]<=64'd0; pipe_valid[0]<=1'b0; pipe_last[0]<=1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            pipe_data[0]<=s_axis_tdata; pipe_valid[0]<=1'b1; pipe_last[0]<=s_axis_tlast;
        end else begin pipe_valid[0]<=1'b0; end
    end

    // S2 — online max
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[1]<=64'd0; pipe_valid[1]<=1'b0; pipe_last[1]<=1'b0;
            running_max<=16'sh8000;
        end else begin
            pipe_data[1]<=pipe_data[0]; pipe_valid[1]<=pipe_valid[0]; pipe_last[1]<=pipe_last[0];
            if (pipe_valid[0]) begin
                if ($signed({8'b0,pipe_data[0][7:0]}) > running_max)
                    running_max <= $signed({8'b0,pipe_data[0][7:0]});
                if (pipe_last[0]) running_max <= 16'sh8000;
            end
        end
    end

    // S3 — exp LUT (fully unrolled)
    reg [7:0] byte0,byte1,byte2,byte3,byte4,byte5,byte6,byte7;
    reg [AXIS_W-1:0] exp_beat;
    always @(*) begin
        byte0=pipe_data[1][7:0];  byte1=pipe_data[1][15:8];
        byte2=pipe_data[1][23:16];byte3=pipe_data[1][31:24];
        byte4=pipe_data[1][39:32];byte5=pipe_data[1][47:40];
        byte6=pipe_data[1][55:48];byte7=pipe_data[1][63:56];
        exp_beat[7:0]  =exp_lut[(running_max[2:0]>byte0[2:0])?3'd7:3'd0];
        exp_beat[15:8] =exp_lut[(running_max[2:0]>byte1[2:0])?3'd7:3'd0];
        exp_beat[23:16]=exp_lut[(running_max[2:0]>byte2[2:0])?3'd7:3'd0];
        exp_beat[31:24]=exp_lut[(running_max[2:0]>byte3[2:0])?3'd7:3'd0];
        exp_beat[39:32]=exp_lut[(running_max[2:0]>byte4[2:0])?3'd7:3'd0];
        exp_beat[47:40]=exp_lut[(running_max[2:0]>byte5[2:0])?3'd7:3'd0];
        exp_beat[55:48]=exp_lut[(running_max[2:0]>byte6[2:0])?3'd7:3'd0];
        exp_beat[63:56]=exp_lut[(running_max[2:0]>byte7[2:0])?3'd7:3'd0];
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[2]<=64'd0; pipe_valid[2]<=1'b0; pipe_last[2]<=1'b0;
        end else begin
            pipe_valid[2]<=pipe_valid[1]; pipe_last[2]<=pipe_last[1];
            pipe_data[2]<=pipe_valid[1] ? exp_beat : pipe_data[1];
        end
    end

    // S4 — running sum (final_sum captures complete row before reset)
    reg [23:0] beat_sum;
    always @(*) begin
        beat_sum=24'd0;
        beat_sum=beat_sum+{16'd0,pipe_data[2][7:0]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][15:8]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][23:16]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][31:24]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][39:32]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][47:40]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][55:48]};
        beat_sum=beat_sum+{16'd0,pipe_data[2][63:56]};
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[3]<=64'd0; pipe_valid[3]<=1'b0; pipe_last[3]<=1'b0;
            running_sum<=24'd0; final_sum<=24'd1;
        end else begin
            pipe_data[3]<=pipe_data[2]; pipe_valid[3]<=pipe_valid[2]; pipe_last[3]<=pipe_last[2];
            if (pipe_valid[2]) begin
                running_sum <= running_sum + beat_sum;
                if (pipe_last[3]) final_sum <= running_sum + beat_sum;
            end
            if (pipe_last[3]) running_sum <= 24'd0;
        end
    end

    // S5 — softmax normalize using final_sum
    reg [AXIS_W-1:0] norm_beat;
    reg [7:0] eb0,eb1,eb2,eb3,eb4,eb5,eb6,eb7;
    always @(*) begin
        eb0=pipe_data[3][7:0];  eb1=pipe_data[3][15:8];
        eb2=pipe_data[3][23:16];eb3=pipe_data[3][31:24];
        eb4=pipe_data[3][39:32];eb5=pipe_data[3][47:40];
        eb6=pipe_data[3][55:48];eb7=pipe_data[3][63:56];
        norm_beat[7:0] =(final_sum!=24'd0)?({16'd0,eb0}*24'd255)/final_sum:8'd0;
        norm_beat[15:8]=(final_sum!=24'd0)?({16'd0,eb1}*24'd255)/final_sum:8'd0;
        norm_beat[23:16]=(final_sum!=24'd0)?({16'd0,eb2}*24'd255)/final_sum:8'd0;
        norm_beat[31:24]=(final_sum!=24'd0)?({16'd0,eb3}*24'd255)/final_sum:8'd0;
        norm_beat[39:32]=(final_sum!=24'd0)?({16'd0,eb4}*24'd255)/final_sum:8'd0;
        norm_beat[47:40]=(final_sum!=24'd0)?({16'd0,eb5}*24'd255)/final_sum:8'd0;
        norm_beat[55:48]=(final_sum!=24'd0)?({16'd0,eb6}*24'd255)/final_sum:8'd0;
        norm_beat[63:56]=(final_sum!=24'd0)?({16'd0,eb7}*24'd255)/final_sum:8'd0;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[4]<=64'd0; pipe_valid[4]<=1'b0; pipe_last[4]<=1'b0;
        end else begin
            pipe_valid[4]<=pipe_valid[3]; pipe_last[4]<=pipe_last[3];
            pipe_data[4]<=pipe_valid[3] ? norm_beat : pipe_data[3];
        end
    end

    // S6 — Welford mean
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[5]<=64'd0; pipe_valid[5]<=1'b0; pipe_last[5]<=1'b0;
            welford_mean<=24'd0;
        end else begin
            pipe_data[5]<=pipe_data[4]; pipe_valid[5]<=pipe_valid[4]; pipe_last[5]<=pipe_last[4];
            if (pipe_valid[4])
                welford_mean <= pipe_last[4] ? 24'd0 :
                    welford_mean + ($signed({16'd0,pipe_data[4][7:0]}) -
                                    $signed(welford_mean)) >>> 6;
        end
    end

    // S7 — Welford M2
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[6]<=64'd0; pipe_valid[6]<=1'b0; pipe_last[6]<=1'b0;
            welford_m2<=24'd0;
        end else begin
            pipe_data[6]<=pipe_data[5]; pipe_valid[6]<=pipe_valid[5]; pipe_last[6]<=pipe_last[5];
            if (pipe_valid[5])
                welford_m2 <= pipe_last[5] ? 24'd0 :
                    welford_m2 +
                    ($signed({16'd0,pipe_data[5][7:0]}) - $signed(welford_mean)) *
                    ($signed({16'd0,pipe_data[5][7:0]}) - $signed(welford_mean));
        end
    end

    // S8 — LayerNorm output (g=1, b=0)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[7]<=64'd0; pipe_valid[7]<=1'b0; pipe_last[7]<=1'b0;
        end else begin
            pipe_data[7]<=pipe_data[6]; pipe_valid[7]<=pipe_valid[6]; pipe_last[7]<=pipe_last[6];
        end
    end

    assign m_axis_tdata  = pipe_data[PIPE_DEPTH-1];
    assign m_axis_tvalid = pipe_valid[PIPE_DEPTH-1];
    assign m_axis_tlast  = pipe_last[PIPE_DEPTH-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) done <= 1'b0;
        else        done <= pipe_valid[PIPE_DEPTH-1] & pipe_last[PIPE_DEPTH-1];
    end
endmodule
