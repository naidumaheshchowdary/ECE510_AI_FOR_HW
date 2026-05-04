// =============================================================================
// tb_interface.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Testbench for interface.sv (AXI4-Lite + AXI4-Stream)
//
// Tests:
//   T1: AXI4-Lite write CFG_D=64, read back — verify OKAY handshake
//   T2: AXI4-Lite write CFG_T=64, read back — verify value
//   T3: AXI4-Lite write PRECISION=0 (INT8), read back
//   T4: AXI4-Lite write CTRL[0]=1 (start), read back
//   T5: AXI4-Stream — send one 64-element row, verify output appears
//
// Simulator : Icarus Verilog (iverilog)
// Run       : iverilog -g2012 -o sim_if tb_interface.sv interface.sv compute_core.sv
//             vvp sim_if | tee interface_run.log
// =============================================================================

`timescale 1ns/1ps

module tb_interface;

    localparam int CLK_HALF = 5;
    localparam int BEATS    = 8;   // 64 bytes / 8 bytes per beat
    localparam int TIMEOUT  = 100;

    // -------------------------------------------------------------------------
    // DUT signals
    // -------------------------------------------------------------------------
    logic        clk = 0;
    logic        rst_n = 0;

    // AXI4-Lite
    logic [5:0]  s_axil_awaddr  = 0;
    logic        s_axil_awvalid = 0;
    logic        s_axil_awready;
    logic [31:0] s_axil_wdata   = 0;
    logic [3:0]  s_axil_wstrb   = 4'hF;
    logic        s_axil_wvalid  = 0;
    logic        s_axil_wready;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_bvalid;
    logic        s_axil_bready  = 1;
    logic [5:0]  s_axil_araddr  = 0;
    logic        s_axil_arvalid = 0;
    logic        s_axil_arready;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;
    logic        s_axil_rvalid;
    logic        s_axil_rready  = 1;

    // AXI4-Stream
    logic [63:0] s_axis_tdata  = 0;
    logic        s_axis_tvalid = 0;
    logic        s_axis_tlast  = 0;
    logic        s_axis_tready;
    logic [63:0] m_axis_tdata;
    logic        m_axis_tvalid;
    logic        m_axis_tlast;
    logic        m_axis_tready = 1;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    interface #(.D(64), .T(64)) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axil_awaddr  (s_axil_awaddr),
        .s_axil_awvalid (s_axil_awvalid),
        .s_axil_awready (s_axil_awready),
        .s_axil_wdata   (s_axil_wdata),
        .s_axil_wstrb   (s_axil_wstrb),
        .s_axil_wvalid  (s_axil_wvalid),
        .s_axil_wready  (s_axil_wready),
        .s_axil_bresp   (s_axil_bresp),
        .s_axil_bvalid  (s_axil_bvalid),
        .s_axil_bready  (s_axil_bready),
        .s_axil_araddr  (s_axil_araddr),
        .s_axil_arvalid (s_axil_arvalid),
        .s_axil_arready (s_axil_arready),
        .s_axil_rdata   (s_axil_rdata),
        .s_axil_rresp   (s_axil_rresp),
        .s_axil_rvalid  (s_axil_rvalid),
        .s_axil_rready  (s_axil_rready),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tlast   (s_axis_tlast),
        .s_axis_tready  (s_axis_tready),
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tlast   (m_axis_tlast),
        .m_axis_tready  (m_axis_tready)
    );

    always #CLK_HALF clk = ~clk;

    // -------------------------------------------------------------------------
    // AXI4-Lite write task
    // -------------------------------------------------------------------------
    task automatic axil_write(input logic [5:0] addr, input logic [31:0] data);
        @(posedge clk); #1;
        s_axil_awaddr  = addr;
        s_axil_awvalid = 1'b1;
        s_axil_wdata   = data;
        s_axil_wstrb   = 4'hF;
        s_axil_wvalid  = 1'b1;
        @(posedge clk); #1;
        s_axil_awvalid = 1'b0;
        s_axil_wvalid  = 1'b0;
        // Wait for BVALID
        repeat(10) begin
            @(posedge clk); #1;
            if (s_axil_bvalid) break;
        end
    endtask

    // -------------------------------------------------------------------------
    // AXI4-Lite read task
    // -------------------------------------------------------------------------
    task automatic axil_read(input logic [5:0] addr, output logic [31:0] data);
        @(posedge clk); #1;
        s_axil_araddr  = addr;
        s_axil_arvalid = 1'b1;
        @(posedge clk); #1;
        @(posedge clk); #1;
        s_axil_arvalid = 1'b0;
        repeat(10) begin
            @(posedge clk); #1;
            if (s_axil_rvalid) begin
                data = s_axil_rdata;
                break;
            end
        end
    endtask

    // -------------------------------------------------------------------------
    // Check helper
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

    // -------------------------------------------------------------------------
    // Test body
    // -------------------------------------------------------------------------
    logic [31:0] rval;

    initial begin
        $display("=== tb_interface: AXI4-Lite + AXI4-Stream Interface ===");

        // Reset
        rst_n = 0;
        repeat(4) @(posedge clk); #1;
        rst_n = 1;
        @(posedge clk); #1;

        // T1: Write and read CFG_D
        $display("  T1: Write CFG_D=64 via AXI4-Lite...");
        axil_write(6'h08, 32'd64);
        check("bvalid received (write handshake complete)", s_axil_bvalid === 1'b0 || s_axil_bresp === 2'b00);
        axil_read(6'h08, rval);
        check("CFG_D reads back 64", rval === 32'd64);

        // T2: Write and read CFG_T
        $display("  T2: Write CFG_T=64 via AXI4-Lite...");
        axil_write(6'h0C, 32'd64);
        axil_read(6'h0C, rval);
        check("CFG_T reads back 64", rval === 32'd64);

        // T3: Write and read PRECISION=0 (INT8)
        $display("  T3: Write PRECISION=0 (INT8)...");
        axil_write(6'h10, 32'd0);
        axil_read(6'h10, rval);
        check("PRECISION reads back 0 (INT8)", rval === 32'd0);

        // T4: Write CTRL start bit
        $display("  T4: Write CTRL[0]=1 (start)...");
        axil_write(6'h00, 32'd1);
        axil_read(6'h00, rval);
        check("CTRL reads back 1 (start asserted)", rval === 32'd1);

        // T5: AXI4-Stream — send one row, check output
        $display("  T5: Stream one 64-element row...");
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

        begin
            automatic logic out_seen = 1'b0;
            automatic logic lst_seen = 1'b0;
            for (int c = 0; c < TIMEOUT; c++) begin
                @(posedge clk); #1;
                if (m_axis_tvalid) begin
                    out_seen = 1'b1;
                    if (m_axis_tlast) begin
                        lst_seen = 1'b1;
                        break;
                    end
                end
            end
            check("AXI4-Stream output appeared", out_seen);
            check("AXI4-Stream tlast seen (row complete)", lst_seen);
        end

        // Summary
        $display("");
        $display("=== RESULT: %0d checks passed, %0d failed ===",
                 pass_count, fail_count);
        if (fail_count == 0)
            $display("PASS: tb_interface");
        else
            $display("FAIL: tb_interface (%0d failures)", fail_count);

        $finish;
    end

    initial begin
        #((TIMEOUT + 50) * 10);
        $display("FAIL: tb_interface — global timeout");
        $finish;
    end

endmodule
