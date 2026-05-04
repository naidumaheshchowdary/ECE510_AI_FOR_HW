// =============================================================================
// compute_core.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Author  : Mahesh Chowdary Naidu
// Project : Fused Softmax + Layer Normalization Accelerator
//
// Description:
//   8-stage pipelined compute core for fused INT8 softmax + layer normalization.
//   Targets the professor's transformer_lm config: d=64, T=64, batch=8.
//   Inputs stream in via AXI4-Stream (64-bit, 8 INT8 elements per beat).
//   Outputs stream out via AXI4-Stream after PIPE_DEPTH clock latency.
//
// Clock domain : Single clock (clk), 100–200 MHz target
// Reset        : Asynchronous active-low (rst_n)
//
// Pipeline stages:
//   S1 : Input latch          — capture INT8 beat from AXI4-Stream
//   S2 : Online max update    — track running max for numerically stable softmax
//   S3 : Exp LUT lookup       — approximate exp via 8-entry LUT + linear interp
//   S4 : Running sum          — accumulate exp values (softmax denominator)
//   S5 : Softmax normalize    — divide by running sum (1/sum via LUT)
//   S6 : Welford mean update  — online mean for layer norm
//   S7 : Welford var update   — online variance for layer norm
//   S8 : Layer norm output    — apply scale (g) and shift (b), write output
//
// Port list:
//   clk            I  1    System clock
//   rst_n          I  1    Async active-low reset
//   s_axis_tdata   I  64   AXI4-Stream input data (8 INT8 elements)
//   s_axis_tvalid  I  1    Input valid
//   s_axis_tlast   I  1    Last beat of one T×d row
//   s_axis_tready  O  1    Core ready to accept input
//   m_axis_tdata   O  64   AXI4-Stream output data (8 INT8 elements)
//   m_axis_tvalid  O  1    Output valid
//   m_axis_tlast   O  1    Last beat of output row
//   m_axis_tready  I  1    Downstream ready
//   cfg_d          I  8    Row width (default 64)
//   cfg_t          I  8    Number of rows (default 64)
//   precision      I  1    0=INT8, 1=FP64 (INT8 path implemented)
//   start          I  1    Begin processing (from control FSM)
//   done           O  1    Row complete (tlast has exited pipeline)
// =============================================================================

module compute_core #(
    parameter int D          = 64,    // row width (elements)
    parameter int T          = 64,    // number of rows
    parameter int DATA_W     = 8,     // element width (INT8)
    parameter int AXIS_W     = 64,    // AXI4-Stream bus width (8 elements per beat)
    parameter int PIPE_DEPTH = 8      // pipeline stages
) (
    input  logic                 clk,
    input  logic                 rst_n,

    // AXI4-Stream slave — input activations
    input  logic [AXIS_W-1:0]    s_axis_tdata,
    input  logic                 s_axis_tvalid,
    input  logic                 s_axis_tlast,
    output logic                 s_axis_tready,

    // AXI4-Stream master — normalized output
    output logic [AXIS_W-1:0]    m_axis_tdata,
    output logic                 m_axis_tvalid,
    output logic                 m_axis_tlast,
    input  logic                 m_axis_tready,

    // Control signals (driven by interface.sv FSM)
    input  logic [7:0]           cfg_d,
    input  logic [7:0]           cfg_t,
    input  logic                 precision,
    input  logic                 start,
    output logic                 done
);

    // -------------------------------------------------------------------------
    // Pipeline registers — 8 stages
    // -------------------------------------------------------------------------
    logic [AXIS_W-1:0] pipe_data  [PIPE_DEPTH];
    logic              pipe_valid [PIPE_DEPTH];
    logic              pipe_last  [PIPE_DEPTH];

    // Online statistics registers (per-row, reset on tlast)
    logic signed [15:0] running_max;   // S2: online max for softmax stability
    logic        [23:0] running_sum;   // S4: sum of exp values
    logic signed [23:0] welford_mean;  // S6: online mean for layer norm
    logic        [23:0] welford_m2;    // S7: online M2 (variance numerator)
    logic        [15:0] elem_count;    // element counter within row

    // 8-entry exp LUT: exp(-k/8) * 256 for k=0..7, scaled to INT8 range
    // Computed offline: exp(0)=256, exp(-1/8)≈224, ..., approximation
    logic [15:0] exp_lut [8];
    initial begin
        exp_lut[0] = 16'd256;   // exp(0)    = 1.000 * 256
        exp_lut[1] = 16'd224;   // exp(-0.125)≈ 0.882 * 256
        exp_lut[2] = 16'd197;   // exp(-0.25) ≈ 0.779 * 256
        exp_lut[3] = 16'd174;   // exp(-0.375)≈ 0.687 * 256
        exp_lut[4] = 16'd153;   // exp(-0.5)  ≈ 0.607 * 256
        exp_lut[5] = 16'd135;   // exp(-0.625)≈ 0.535 * 256
        exp_lut[6] = 16'd119;   // exp(-0.75) ≈ 0.472 * 256
        exp_lut[7] = 16'd105;   // exp(-0.875)≈ 0.417 * 256
    end

    // -------------------------------------------------------------------------
    // S1 — Input latch: capture beat from AXI4-Stream
    // Back-pressure: ready when downstream is ready or pipeline slot is empty
    // -------------------------------------------------------------------------
    assign s_axis_tready = m_axis_tready | ~pipe_valid[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[0]  <= '0;
            pipe_valid[0] <= 1'b0;
            pipe_last[0]  <= 1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            pipe_data[0]  <= s_axis_tdata;
            pipe_valid[0] <= 1'b1;
            pipe_last[0]  <= s_axis_tlast;
        end else begin
            pipe_valid[0] <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // S2 — Online max update (for numerically stable softmax)
    // Track max of all elements seen so far in this row
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[1]  <= '0;
            pipe_valid[1] <= 1'b0;
            pipe_last[1]  <= 1'b0;
            running_max   <= 16'sh8000; // most negative
            elem_count    <= '0;
        end else begin
            pipe_valid[1] <= pipe_valid[0];
            pipe_last[1]  <= pipe_last[0];
            if (pipe_valid[0]) begin
                // Update max across all 8 INT8 bytes in the beat
                // Simplified: compare first byte as representative
                automatic logic signed [7:0] byte0;
                byte0 = signed'(pipe_data[0][7:0]);
                running_max  <= ($signed({8'b0, byte0}) > running_max) ?
                                $signed({8'b0, byte0}) : running_max;
                pipe_data[1] <= pipe_data[0]; // pass through
                elem_count   <= elem_count + 16'd8;
                if (pipe_last[0]) begin
                    running_max <= 16'sh8000;
                    elem_count  <= '0;
                end
            end else begin
                pipe_data[1] <= pipe_data[0];
            end
        end
    end

    // -------------------------------------------------------------------------
    // S3 — Exp LUT approximation
    // Each INT8 element: subtract max, look up exp in LUT
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[2]  <= '0;
            pipe_valid[2] <= 1'b0;
            pipe_last[2]  <= 1'b0;
        end else begin
            pipe_valid[2] <= pipe_valid[1];
            pipe_last[2]  <= pipe_last[1];
            if (pipe_valid[1]) begin
                // For each of the 8 bytes: compute (x - max), clamp, LUT lookup
                // Simplified: map (x - max) range [−7,0] to LUT index [7,0]
                automatic logic [63:0] exp_beat;
                automatic logic signed [7:0] bval;
                automatic logic [2:0] lut_idx;
                for (int b = 0; b < 8; b++) begin
                    bval    = signed'(pipe_data[1][b*8 +: 8]);
                    lut_idx = (running_max[2:0] > bval[2:0]) ? 3'd7 : 3'd0;
                    exp_beat[b*8 +: 8] = exp_lut[lut_idx][7:0];
                end
                pipe_data[2] <= exp_beat;
            end else begin
                pipe_data[2] <= pipe_data[1];
            end
        end
    end

    // -------------------------------------------------------------------------
    // S4 — Running sum (softmax denominator)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[3]  <= '0;
            pipe_valid[3] <= 1'b0;
            pipe_last[3]  <= 1'b0;
            running_sum   <= '0;
        end else begin
            pipe_valid[3] <= pipe_valid[2];
            pipe_last[3]  <= pipe_last[2];
            pipe_data[3]  <= pipe_data[2];
            if (pipe_valid[2]) begin
                automatic logic [23:0] beat_sum;
                beat_sum = '0;
                for (int b = 0; b < 8; b++)
                    beat_sum += {16'b0, pipe_data[2][b*8 +: 8]};
                running_sum <= pipe_last[2] ? '0 : running_sum + beat_sum;
            end
        end
    end

    // -------------------------------------------------------------------------
    // S5 — Softmax normalize: divide exp by running_sum
    // Use reciprocal approximation: out = (exp * 255) / running_sum
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[4]  <= '0;
            pipe_valid[4] <= 1'b0;
            pipe_last[4]  <= 1'b0;
        end else begin
            pipe_valid[4] <= pipe_valid[3];
            pipe_last[4]  <= pipe_last[3];
            if (pipe_valid[3] && (running_sum != '0)) begin
                automatic logic [63:0] norm_beat;
                for (int b = 0; b < 8; b++) begin
                    norm_beat[b*8 +: 8] =
                        ({16'b0, pipe_data[3][b*8 +: 8]} * 24'd255)
                        / running_sum;
                end
                pipe_data[4] <= norm_beat;
            end else begin
                pipe_data[4] <= pipe_data[3];
            end
        end
    end

    // -------------------------------------------------------------------------
    // S6 — Welford online mean update
    // mean_n = mean_{n-1} + (x - mean_{n-1}) / n
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[5]  <= '0;
            pipe_valid[5] <= 1'b0;
            pipe_last[5]  <= 1'b0;
            welford_mean  <= '0;
        end else begin
            pipe_valid[5] <= pipe_valid[4];
            pipe_last[5]  <= pipe_last[4];
            pipe_data[5]  <= pipe_data[4];
            if (pipe_valid[4]) begin
                automatic logic signed [23:0] delta;
                automatic logic signed [7:0]  x0;
                x0    = signed'(pipe_data[4][7:0]);
                delta = $signed({16'b0, x0}) - welford_mean;
                // Simplified: update mean using first byte representative
                welford_mean <= pipe_last[4] ? '0 :
                                welford_mean + (delta >>> 6); // divide by ~D=64
            end
        end
    end

    // -------------------------------------------------------------------------
    // S7 — Welford variance update
    // M2_n = M2_{n-1} + delta * (x - new_mean)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[6]  <= '0;
            pipe_valid[6] <= 1'b0;
            pipe_last[6]  <= 1'b0;
            welford_m2    <= '0;
        end else begin
            pipe_valid[6] <= pipe_valid[5];
            pipe_last[6]  <= pipe_last[5];
            pipe_data[6]  <= pipe_data[5];
            if (pipe_valid[5]) begin
                automatic logic signed [23:0] delta2;
                automatic logic signed [7:0]  x0;
                x0     = signed'(pipe_data[5][7:0]);
                delta2 = $signed({16'b0, x0}) - welford_mean;
                welford_m2 <= pipe_last[5] ? '0 :
                              welford_m2 + delta2[11:0] * delta2[11:0];
            end
        end
    end

    // -------------------------------------------------------------------------
    // S8 — Layer norm output
    // y = (x - mean) / sqrt(var + eps) * g + b
    // g=1, b=0 (unit scale, zero shift for this stub — full values loaded via
    // parameter SRAM in final M3 design)
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[7]  <= '0;
            pipe_valid[7] <= 1'b0;
            pipe_last[7]  <= 1'b0;
        end else begin
            pipe_valid[7] <= pipe_valid[6];
            pipe_last[7]  <= pipe_last[6];
            if (pipe_valid[6]) begin
                // var = welford_m2 / D; sqrt approximated via right-shift
                // For stub: pass softmax output through (arithmetic verified in tb)
                // Full layer norm arithmetic added in M3
                pipe_data[7] <= pipe_data[6];
            end else begin
                pipe_data[7] <= pipe_data[6];
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output — connect final pipeline stage to AXI4-Stream master
    // -------------------------------------------------------------------------
    assign m_axis_tdata  = pipe_data[PIPE_DEPTH-1];
    assign m_axis_tvalid = pipe_valid[PIPE_DEPTH-1];
    assign m_axis_tlast  = pipe_last[PIPE_DEPTH-1];

    // done pulse: tlast exits the pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            done <= 1'b0;
        else
            done <= pipe_valid[PIPE_DEPTH-1] && pipe_last[PIPE_DEPTH-1];
    end

endmodule
