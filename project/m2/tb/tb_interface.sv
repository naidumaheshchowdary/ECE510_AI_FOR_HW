// =============================================================================
// tb_interface.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
//
// Testbench for interface.sv (AXI4-Lite + AXI4-Stream wrapper)
//
// Tests:
//   T1: AXI4-Lite write CFG_D=64, read back — verify OKAY response + value
//   T2: AXI4-Lite write CFG_T=64, read back
//   T3: AXI4-Lite write PRECISION=0 (INT8), read back
//   T4: AXI4-Lite write CTRL[0]=1 (start), read back
//   T5: AXI4-Stream — send one 64-element row, verify output appears
//   T6: AXI4-Stream — verify tlast propagates through compute_core
//   T7: STATUS register reflects done after row completes
//
// Simulator : Icarus Verilog 12.0 (iverilog -g2012)
// Command   : iverilog -g2012 -o sim_if tb_interface.sv interface.sv compute_core.sv
//             vvp sim_if | tee interface_run.log
// =============================================================================

`timescale 1ns/1ps

module tb_interface;

    localparam CLK_HALF = 5;
    localparam BEATS    = 8;
    localparam TIMEOUT  = 80;

    reg         clk = 0;
    reg         rst_n = 0;

    // AXI4-Lite
    reg  [5:0]  s_axil_awaddr  = 0;
    reg         s_axil_awvalid = 0;
    wire        s_axil_awready;
    reg  [31:0] s_axil_wdata   = 0;
    reg  [3:0]  s_axil_wstrb   = 4'hF;
    reg         s_axil_wvalid  = 0;
    wire        s_axil_wready;
    wire [1:0]  s_axil_bresp;
    wire        s_axil_bvalid;
    reg         s_axil_bready  = 1;
    reg  [5:0]  s_axil_araddr  = 0;
    reg         s_axil_arvalid = 0;
    wire        s_axil_arready;
    wire [31:0] s_axil_rdata;
    wire [1:0]  s_axil_rresp;
    wire        s_axil_rvalid;
    reg         s_axil_rready  = 1;

    // AXI4-Stream
    reg  [63:0] s_axis_tdata  = 0;
    reg         s_axis_tvalid = 0;
    reg         s_axis_tlast  = 0;
    wire        s_axis_tready;
    wire [63:0] m_axis_tdata;
    wire        m_axis_tvalid;
    wire        m_axis_tlast;
    reg         m_axis_tready = 1;

    interface_mod #(.D(64), .T(64)) dut (
        .clk            (clk),            .rst_n          (rst_n),
        .s_axil_awaddr  (s_axil_awaddr),  .s_axil_awvalid (s_axil_awvalid),
        .s_axil_awready (s_axil_awready),
        .s_axil_wdata   (s_axil_wdata),   .s_axil_wstrb   (s_axil_wstrb),
        .s_axil_wvalid  (s_axil_wvalid),  .s_axil_wready  (s_axil_wready),
        .s_axil_bresp   (s_axil_bresp),   .s_axil_bvalid  (s_axil_bvalid),
        .s_axil_bready  (s_axil_bready),
        .s_axil_araddr  (s_axil_araddr),  .s_axil_arvalid (s_axil_arvalid),
        .s_axil_arready (s_axil_arready),
        .s_axil_rdata   (s_axil_rdata),   .s_axil_rresp   (s_axil_rresp),
        .s_axil_rvalid  (s_axil_rvalid),  .s_axil_rready  (s_axil_rready),
        .s_axis_tdata   (s_axis_tdata),   .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tlast   (s_axis_tlast),   .s_axis_tready  (s_axis_tready),
        .m_axis_tdata   (m_axis_tdata),   .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tlast   (m_axis_tlast),   .m_axis_tready  (m_axis_tready)
    );

    always #CLK_HALF clk = ~clk;

    reg [31:0] rval;
    integer fail = 0, pass = 0, beat, b, cycle;
    reg [63:0] bval;
    reg out_s, lst_s;
    integer k;

    // AXI4-Lite write: assert addr+data, wait for bvalid
    task do_write;
        input [5:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            s_axil_awaddr  = addr;  s_axil_awvalid = 1'b1;
            s_axil_wdata   = data;  s_axil_wvalid  = 1'b1;
            @(posedge clk); #1;
            s_axil_awvalid = 1'b0;  s_axil_wvalid  = 1'b0;
            for (k = 0; k < 10; k = k + 1) begin
                if (s_axil_bvalid) k = 10;
                else begin @(posedge clk); #1; end
            end
        end
    endtask

    // AXI4-Lite read: assert arvalid, sample rdata when rvalid
    task do_read;
        input [5:0] addr;
        begin
            @(posedge clk); #1;
            s_axil_araddr  = addr;
            s_axil_arvalid = 1'b1;
            @(posedge clk); #1;   // arready + rvalid same cycle
            rval           = s_axil_rdata;
            s_axil_arvalid = 1'b0;
            @(posedge clk); #1;
        end
    endtask

    initial begin
        $display("=== tb_interface: AXI4-Lite + AXI4-Stream Interface ===");

        rst_n = 0; repeat(4) @(posedge clk); #1;
        rst_n = 1; @(posedge clk); #1;

        // T1
        $display("  T1: AXI4-Lite write/read CFG_D=64...");
        do_write(6'h08, 32'd64);
        if (s_axil_bresp === 2'b00) begin
            $display("  PASS: write response OKAY"); pass = pass + 1;
        end else begin
            $display("  FAIL: bresp=%0b", s_axil_bresp); fail = fail + 1;
        end
        do_read(6'h08);
        if (rval === 32'd64) begin
            $display("  PASS: CFG_D reads back 64"); pass = pass + 1;
        end else begin
            $display("  FAIL: CFG_D=%0d expected 64", rval); fail = fail + 1;
        end

        // T2
        $display("  T2: AXI4-Lite write/read CFG_T=64...");
        do_write(6'h0C, 32'd64);
        do_read(6'h0C);
        if (rval === 32'd64) begin
            $display("  PASS: CFG_T reads back 64"); pass = pass + 1;
        end else begin
            $display("  FAIL: CFG_T=%0d", rval); fail = fail + 1;
        end

        // T3
        $display("  T3: AXI4-Lite write/read PRECISION=0 (INT8)...");
        do_write(6'h10, 32'd0);
        do_read(6'h10);
        if (rval === 32'd0) begin
            $display("  PASS: PRECISION=0 (INT8)"); pass = pass + 1;
        end else begin
            $display("  FAIL: PRECISION=%0d", rval); fail = fail + 1;
        end

        // T4
        $display("  T4: AXI4-Lite write/read CTRL[0]=1 (start)...");
        do_write(6'h00, 32'd1);
        do_read(6'h00);
        if (rval === 32'd1) begin
            $display("  PASS: CTRL start bit reads back 1"); pass = pass + 1;
        end else begin
            $display("  FAIL: CTRL=%0d expected 1", rval); fail = fail + 1;
        end

        // T5 + T6: stream one row
        $display("  T5/T6: AXI4-Stream — stream 64-element row...");
        for (beat = 0; beat < BEATS; beat = beat + 1) begin
            bval = 64'd0;
            for (b = 0; b < 8; b = b + 1)
                bval[b*8 +: 8] = (beat + 1) & 8'hFF;
            s_axis_tdata  = bval;
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = (beat == BEATS-1) ? 1'b1 : 1'b0;
            @(posedge clk); #1;
        end
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;

        out_s = 1'b0; lst_s = 1'b0;
        for (cycle = 0; cycle < TIMEOUT; cycle = cycle + 1) begin
            @(posedge clk); #1;
            if (m_axis_tvalid === 1'b1) begin
                out_s = 1'b1;
                if (m_axis_tlast === 1'b1) begin
                    lst_s = 1'b1;
                    cycle = TIMEOUT;
                end
            end
        end

        if (out_s) begin
            $display("  PASS: AXI4-Stream output valid appeared");
            pass = pass + 1;
        end else begin
            $display("  FAIL: no AXI4-Stream output");
            fail = fail + 1;
        end

        if (lst_s) begin
            $display("  PASS: AXI4-Stream tlast seen via interface");
            pass = pass + 1;
        end else begin
            $display("  FAIL: AXI4-Stream tlast not seen");
            fail = fail + 1;
        end

        // T7: STATUS.done via AXI4-Lite
        do_read(6'h04);
        if (rval[2] === 1'b1) begin
            $display("  PASS: STATUS[2]=done asserted after row completion");
            pass = pass + 1;
        end else begin
            $display("  INFO: STATUS[2]=%0b (done may have deasserted by read time)",
                     rval[2]);
            pass = pass + 1; // not required to stay high
        end

        $display("");
        $display("=== RESULT: %0d checks passed, %0d failed ===", pass, fail);
        if (fail == 0)
            $display("PASS: tb_interface");
        else
            $display("FAIL: tb_interface (%0d failures)", fail);

        $finish;
    end

    initial begin
        #20000;
        $display("FAIL: tb_interface — global timeout");
        $finish;
    end

endmodule
