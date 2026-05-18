// =============================================================================
// compute_core.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Author  : Mahesh Chowdary Naidu
// Project : Fused Softmax + Layer Normalization Accelerator
//
// Description:
//   8-stage pipelined compute core for fused INT8 softmax + layer normalization.
//   Targets professor's transformer_lm config: d=64, T=64, batch=8.
//   Inputs stream via AXI4-Stream (64-bit, 8 INT8 elements per beat).
//   Outputs stream via AXI4-Stream after PIPE_DEPTH=8 clock cycles latency.
//
// Clock domain : Single clock (clk), 100-200 MHz target on SKY130
// Reset        : Asynchronous active-low (rst_n) — consistent across all always blocks
//
// Pipeline stages:
//   S1 : Input latch       — capture INT8 beat, back-pressure via tready
//   S2 : Online max        — running max for numerically stable softmax
//   S3 : Exp LUT           — 8-entry exp approximation, fully unrolled per byte
//   S4 : Running sum       — softmax denominator accumulation (24-bit)
//   S5 : Softmax normalize — (exp * 255) / running_sum per byte, unrolled
//   S6 : Welford mean      — online mean update for layer normalization
//   S7 : Welford variance  — online M2 update (variance numerator)
//   S8 : LayerNorm output  — g*x_hat + b (g=1, b=0 stub; full arithmetic in M3)
//
// Port list (name | direction | width | purpose):
//   clk            I   1    System clock — single clock domain
//   rst_n          I   1    Asynchronous active-low reset
//   s_axis_tdata   I  64    AXI4-Stream input: 8 x INT8 activation elements per beat
//   s_axis_tvalid  I   1    AXI4-Stream input valid
//   s_axis_tlast   I   1    AXI4-Stream last beat of input row
//   s_axis_tready  O   1    AXI4-Stream back-pressure (core ready to accept)
//   m_axis_tdata   O  64    AXI4-Stream output: 8 x INT8 normalized elements per beat
//   m_axis_tvalid  O   1    AXI4-Stream output valid
//   m_axis_tlast   O   1    AXI4-Stream last beat of output row
//   m_axis_tready  I   1    AXI4-Stream downstream ready
//   cfg_d          I   8    Row width in elements (default 64)
//   cfg_t          I   8    Number of rows (default 64)
//   precision      I   1    0=INT8 (implemented), 1=FP64 (reserved for M3)
//   start          I   1    Start processing — from AXI4-Lite CTRL[0] via interface
//   done           O   1    Done pulse: high one cycle after tlast exits pipeline
// =============================================================================

`timescale 1ns/1ps
module synth_top #(
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

    // -------------------------------------------------------------------------
    // Pipeline registers — 8 stages
    // -------------------------------------------------------------------------
    reg [AXIS_W-1:0] pipe_data  [0:PIPE_DEPTH-1];
    reg              pipe_valid [0:PIPE_DEPTH-1];
    reg              pipe_last  [0:PIPE_DEPTH-1];

    // Per-row online statistics
    reg signed [15:0] running_max;   // S2: numerically stable softmax max
    reg        [23:0] running_sum;   // S4: softmax denominator
    reg signed [23:0] welford_mean;  // S6: Welford online mean
    reg        [23:0] welford_m2;    // S7: Welford online variance numerator

    // Exp LUT: exp(-k/8)*255 for k=0..7  (synthesizes as 8-entry ROM)
    reg [7:0] exp_lut [0:7];

    initial begin
        exp_lut[0] = 8'd255; exp_lut[1] = 8'd224;
        exp_lut[2] = 8'd197; exp_lut[3] = 8'd174;
        exp_lut[4] = 8'd153; exp_lut[5] = 8'd135;
        exp_lut[6] = 8'd119; exp_lut[7] = 8'd105;
    end

    // -------------------------------------------------------------------------
    // S1 — Input latch + AXI4-Stream back-pressure
    // tready: asserted when pipeline slot 0 is empty OR downstream is ready
    // -------------------------------------------------------------------------
    always @(*) s_axis_tready = m_axis_tready | ~pipe_valid[0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[0] <= 64'd0; pipe_valid[0] <= 1'b0; pipe_last[0] <= 1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            pipe_data[0] <= s_axis_tdata; pipe_valid[0] <= 1'b1; pipe_last[0] <= s_axis_tlast;
        end else begin
            pipe_valid[0] <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // S2 — Online max (first byte representative; full byte-wise in M3)
    // Reset running_max on tlast to prepare for next row
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[1] <= 64'd0; pipe_valid[1] <= 1'b0;
            pipe_last[1] <= 1'b0;  running_max   <= 16'sh8000;
        end else begin
            pipe_data[1] <= pipe_data[0];
            pipe_valid[1] <= pipe_valid[0];
            pipe_last[1]  <= pipe_last[0];
            if (pipe_valid[0]) begin
                if ($signed({8'b0, pipe_data[0][7:0]}) > running_max)
                    running_max <= $signed({8'b0, pipe_data[0][7:0]});
                if (pipe_last[0]) running_max <= 16'sh8000;
            end
        end
    end

    // -------------------------------------------------------------------------
    // S3 — Exp LUT lookup
    // Fully unrolled to avoid variable part-select (unsupported in Icarus 12)
    // LUT index: 7 if element < running_max, 0 if at max
    // -------------------------------------------------------------------------
    reg [7:0]  byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7;
    reg [AXIS_W-1:0] exp_beat;

    always @(*) begin
        byte0 = pipe_data[1][ 7: 0]; byte1 = pipe_data[1][15: 8];
        byte2 = pipe_data[1][23:16]; byte3 = pipe_data[1][31:24];
        byte4 = pipe_data[1][39:32]; byte5 = pipe_data[1][47:40];
        byte6 = pipe_data[1][55:48]; byte7 = pipe_data[1][63:56];
        exp_beat[ 7: 0] = exp_lut[(running_max[2:0] > byte0[2:0]) ? 3'd7 : 3'd0];
        exp_beat[15: 8] = exp_lut[(running_max[2:0] > byte1[2:0]) ? 3'd7 : 3'd0];
        exp_beat[23:16] = exp_lut[(running_max[2:0] > byte2[2:0]) ? 3'd7 : 3'd0];
        exp_beat[31:24] = exp_lut[(running_max[2:0] > byte3[2:0]) ? 3'd7 : 3'd0];
        exp_beat[39:32] = exp_lut[(running_max[2:0] > byte4[2:0]) ? 3'd7 : 3'd0];
        exp_beat[47:40] = exp_lut[(running_max[2:0] > byte5[2:0]) ? 3'd7 : 3'd0];
        exp_beat[55:48] = exp_lut[(running_max[2:0] > byte6[2:0]) ? 3'd7 : 3'd0];
        exp_beat[63:56] = exp_lut[(running_max[2:0] > byte7[2:0]) ? 3'd7 : 3'd0];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[2] <= 64'd0; pipe_valid[2] <= 1'b0; pipe_last[2] <= 1'b0;
        end else begin
            pipe_valid[2] <= pipe_valid[1]; pipe_last[2] <= pipe_last[1];
            pipe_data[2]  <= pipe_valid[1] ? exp_beat : pipe_data[1];
        end
    end

    // -------------------------------------------------------------------------
    // S4 — Running sum (softmax denominator, 24-bit)
    // Fully unrolled accumulation — no loop variable
    // -------------------------------------------------------------------------
    reg [23:0] beat_sum;
    always @(*) begin
        beat_sum = 24'd0;
        beat_sum = beat_sum + {16'd0, pipe_data[2][ 7: 0]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][15: 8]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][23:16]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][31:24]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][39:32]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][47:40]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][55:48]};
        beat_sum = beat_sum + {16'd0, pipe_data[2][63:56]};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[3] <= 64'd0; pipe_valid[3] <= 1'b0;
            pipe_last[3] <= 1'b0;  running_sum   <= 24'd0;
        end else begin
            pipe_data[3] <= pipe_data[2]; pipe_valid[3] <= pipe_valid[2]; pipe_last[3] <= pipe_last[2];
            if (pipe_valid[2])
                running_sum <= pipe_last[2] ? 24'd0 : running_sum + beat_sum;
        end
    end

    // -------------------------------------------------------------------------
    // S5 — Softmax normalize: out[b] = (exp[b] * 255) / running_sum
    // Fully unrolled — 8 independent divisions
    // -------------------------------------------------------------------------
    reg [AXIS_W-1:0] norm_beat;
    reg [7:0] eb0, eb1, eb2, eb3, eb4, eb5, eb6, eb7;

    always @(*) begin
        eb0 = pipe_data[3][ 7: 0]; eb1 = pipe_data[3][15: 8];
        eb2 = pipe_data[3][23:16]; eb3 = pipe_data[3][31:24];
        eb4 = pipe_data[3][39:32]; eb5 = pipe_data[3][47:40];
        eb6 = pipe_data[3][55:48]; eb7 = pipe_data[3][63:56];
        norm_beat[ 7: 0] = (running_sum != 24'd0) ? ({16'd0,eb0} * 24'd255) / running_sum : 8'd0;
        norm_beat[15: 8] = (running_sum != 24'd0) ? ({16'd0,eb1} * 24'd255) / running_sum : 8'd0;
        norm_beat[23:16] = (running_sum != 24'd0) ? ({16'd0,eb2} * 24'd255) / running_sum : 8'd0;
        norm_beat[31:24] = (running_sum != 24'd0) ? ({16'd0,eb3} * 24'd255) / running_sum : 8'd0;
        norm_beat[39:32] = (running_sum != 24'd0) ? ({16'd0,eb4} * 24'd255) / running_sum : 8'd0;
        norm_beat[47:40] = (running_sum != 24'd0) ? ({16'd0,eb5} * 24'd255) / running_sum : 8'd0;
        norm_beat[55:48] = (running_sum != 24'd0) ? ({16'd0,eb6} * 24'd255) / running_sum : 8'd0;
        norm_beat[63:56] = (running_sum != 24'd0) ? ({16'd0,eb7} * 24'd255) / running_sum : 8'd0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[4] <= 64'd0; pipe_valid[4] <= 1'b0; pipe_last[4] <= 1'b0;
        end else begin
            pipe_valid[4] <= pipe_valid[3]; pipe_last[4] <= pipe_last[3];
            pipe_data[4]  <= pipe_valid[3] ? norm_beat : pipe_data[3];
        end
    end

    // -------------------------------------------------------------------------
    // S6 — Welford online mean: mean += (x - mean) >> 6  (divide by ~D=64)
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[5] <= 64'd0; pipe_valid[5] <= 1'b0;
            pipe_last[5] <= 1'b0;  welford_mean  <= 24'd0;
        end else begin
            pipe_data[5] <= pipe_data[4]; pipe_valid[5] <= pipe_valid[4]; pipe_last[5] <= pipe_last[4];
            if (pipe_valid[4])
                welford_mean <= pipe_last[4] ? 24'd0 :
                    welford_mean + ($signed({16'd0, pipe_data[4][7:0]}) -
                                   $signed(welford_mean)) >>> 6;
        end
    end

    // -------------------------------------------------------------------------
    // S7 — Welford M2: M2 += delta * (x - new_mean)
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[6] <= 64'd0; pipe_valid[6] <= 1'b0;
            pipe_last[6] <= 1'b0;  welford_m2   <= 24'd0;
        end else begin
            pipe_data[6] <= pipe_data[5]; pipe_valid[6] <= pipe_valid[5]; pipe_last[6] <= pipe_last[5];
            if (pipe_valid[5])
                welford_m2 <= pipe_last[5] ? 24'd0 :
                    welford_m2 + ($signed({16'd0, pipe_data[5][7:0]}) - $signed(welford_mean)) *
                                 ($signed({16'd0, pipe_data[5][7:0]}) - $signed(welford_mean));
        end
    end

    // -------------------------------------------------------------------------
    // S8 — LayerNorm output: y = x (g=1, b=0 stub; full scale/shift in M3)
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[7] <= 64'd0; pipe_valid[7] <= 1'b0; pipe_last[7] <= 1'b0;
        end else begin
            pipe_data[7] <= pipe_data[6]; pipe_valid[7] <= pipe_valid[6]; pipe_last[7] <= pipe_last[6];
        end
    end

    // -------------------------------------------------------------------------
    // Output — connect final stage to AXI4-Stream master port
    // -------------------------------------------------------------------------
    assign m_axis_tdata  = pipe_data[PIPE_DEPTH-1];
    assign m_axis_tvalid = pipe_valid[PIPE_DEPTH-1];
    assign m_axis_tlast  = pipe_last[PIPE_DEPTH-1];

    // done: registered pulse one cycle after tlast exits pipeline
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) done <= 1'b0;
        else        done <= pipe_valid[PIPE_DEPTH-1] & pipe_last[PIPE_DEPTH-1];
    end

endmodule

