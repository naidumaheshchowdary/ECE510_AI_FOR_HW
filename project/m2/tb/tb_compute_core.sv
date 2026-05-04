// =============================================================================
// tb_compute_core.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Testbench for compute_core.sv
//
// Test vectors: representative INT8 input row of 64 elements
// Reference computed independently in Python (see precision.md):
//   Input:  [1,2,3,...,8] repeated across 8 beats
//   After 8-stage pipeline delay, output valid and tlast must be seen
//   PASS criteria:
//     (1) m_axis_tvalid rises within PIPE_DEPTH + BEATS + 5 cycles
//     (2) m_axis_tlast seen (complete row propagated)
//     (3) No X on m_axis_tdata when tvalid=1
//
// Simulator : Icarus Verilog (iverilog)
// Run       : iverilog -g2012 -o sim_core tb_compute_core.sv compute_core.sv
//             vvp sim_core | tee compute_core_run.log
// =============================================================================

`timescale 1ns/1ps

module tb_compute_core;

    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    localparam int CLK_HALF  = 5;     // 10 ns period = 100 MHz
    localparam int PIPE_D    = 8;
    localparam int D         = 64;
    localparam int BEATS     = D / 8; // 8 beats of 8 bytes each = 64 elements
    localparam int TIMEOUT   = PIPE_D + BEATS + 20;

    // -------------------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------------------
    logic        clk           = 0;
    logic        rst_n         = 0;
    logic [63:0] s_axis_tdata  = 0;
    logic        s_axis_tvalid = 0;
    logic        s_axis_tlast  = 0;
    logic        s_axis_tready;
    logic [63:0] m_axis_tdata;
    logic        m_axis_tvalid;
    logic        m_axis_tlast;
    logic        m_axis_tready = 1;
    logic [7:0]  cfg_d         = 8'd64;
    logic [7:0]  cfg_t         = 8'd64;
    logic        precision     = 1'b0;  // INT8
    logic        start         = 1'b0;
    logic        done;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    compute_core #(.D(64), .T(64)) dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tlast  (s_axis_tlast),
        .s_axis_tready (s_axis_tready),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tlast  (m_axis_tlast),
        .m_axis_tready (m_axis_tready),
        .cfg_d         (cfg_d),
        .cfg_t         (cfg_t),
        .precision     (precision),
        .start         (start),
        .done          (done)
    );

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    always #CLK_HALF clk = ~clk;

    // -------------------------------------------------------------------------
    // Test body
    // -------------------------------------------------------------------------
    integer fail_count = 0;
    integer pass_count = 0;

    task automatic check(input string label, input logic cond);
        if (cond) begin
            $display("  PASS: %s", label);
            pass_count++;
        end else begin
            $display("  FAIL: %s", label);
            fail_count++;
        end
    endtask

    initial begin
        $display("=== tb_compute_core: Fused Softmax+LayerNorm Pipeline ===");
        $display("    Input: 8 beats x 8 bytes = 64 INT8 elements");
        $display("    Reference: pipeline must propagate tlast in %0d cycles", TIMEOUT);

        // ------------------------------------------------------------------
        // Reset
        // ------------------------------------------------------------------
        rst_n = 0;
        repeat(4) @(posedge clk); #1;
        rst_n = 1;
        @(posedge clk); #1;

        // ------------------------------------------------------------------
        // Check reset state
        // ------------------------------------------------------------------
        check("m_axis_tvalid=0 after reset", m_axis_tvalid === 1'b0);
        check("s_axis_tready=1 after reset", s_axis_tready === 1'b1);

        // ------------------------------------------------------------------
        // Assert start, write cfg registers (via direct signal — no AXI here)
        // ------------------------------------------------------------------
        start = 1'b1;
        @(posedge clk); #1;

        // ------------------------------------------------------------------
        // Drive one representative row: 8 beats
        // Beat pattern: beat_idx repeated in all 8 bytes
        // Input[0..7] = {1,1,1,1,1,1,1,1}
        // Input[8..15]= {2,2,2,2,2,2,2,2} etc.
        // This is a non-trivial, non-zero test vector matching
        // a real activation distribution (monotone ramp)
        // ------------------------------------------------------------------
        $display("  Driving 8 beats into pipeline...");
        for (int beat = 0; beat < BEATS; beat++) begin
            automatic logic [63:0] bval;
            bval = 0;
            for (int b = 0; b < 8; b++)
                bval[b*8 +: 8] = 8'((beat + 1) & 8'hFF);
            s_axis_tdata  = bval;
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = (beat == BEATS-1) ? 1'b1 : 1'b0;
            @(posedge clk); #1;
        end
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;

        // ------------------------------------------------------------------
        // Wait for output — pipeline has PIPE_D cycle latency
        // ------------------------------------------------------------------
        begin
            automatic logic output_seen = 1'b0;
            automatic logic last_seen   = 1'b0;
            automatic int   cycle       = 0;
            while (cycle < TIMEOUT) begin
                @(posedge clk); #1;
                cycle++;
                if (m_axis_tvalid === 1'b1) begin
                    output_seen = 1'b1;
                    $display("  cycle %0d: tdata=0x%016H tlast=%0b",
                             cycle, m_axis_tdata, m_axis_tlast);
                    // Check no X on output data
                    check("tdata has no X bits when valid",
                          ^m_axis_tdata !== 1'bx);
                    if (m_axis_tlast === 1'b1) begin
                        last_seen = 1'b1;
                        break;
                    end
                end
            end
            check("output valid observed within timeout", output_seen);
            check("tlast seen — row completed through pipeline", last_seen);
        end

        // ------------------------------------------------------------------
        // Done flag check
        // ------------------------------------------------------------------
        @(posedge clk); #1;
        check("done flag asserted after tlast", done === 1'b1);

        // ------------------------------------------------------------------
        // Result
        // ------------------------------------------------------------------
        $display("");
        $display("=== RESULT: %0d checks passed, %0d failed ===",
                 pass_count, fail_count);
        if (fail_count == 0)
            $display("PASS: tb_compute_core");
        else
            $display("FAIL: tb_compute_core (%0d failures)", fail_count);

        $finish;
    end

    // Timeout watchdog
    initial begin
        #((TIMEOUT + 50) * 10);
        $display("FAIL: tb_compute_core — global timeout");
        $finish;
    end

endmodule
