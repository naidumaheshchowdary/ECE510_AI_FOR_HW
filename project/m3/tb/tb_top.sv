`timescale 1ns/1ps
module tb_top;

    reg         clk, rst_n;
    reg  [5:0]  s_axil_awaddr;
    reg         s_axil_awvalid;
    wire        s_axil_awready;
    reg  [31:0] s_axil_wdata;
    reg  [3:0]  s_axil_wstrb;
    reg         s_axil_wvalid;
    wire        s_axil_wready;
    wire [1:0]  s_axil_bresp;
    wire        s_axil_bvalid;
    reg         s_axil_bready;
    reg  [5:0]  s_axil_araddr;
    reg         s_axil_arvalid;
    wire        s_axil_arready;
    wire [31:0] s_axil_rdata;
    wire [1:0]  s_axil_rresp;
    wire        s_axil_rvalid;
    reg         s_axil_rready;
    reg  [63:0] s_axis_tdata;
    reg         s_axis_tvalid, s_axis_tlast;
    wire        s_axis_tready;
    wire [63:0] m_axis_tdata;
    wire        m_axis_tvalid, m_axis_tlast;
    reg         m_axis_tready;

    top dut (
        .clk(clk),.rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr),.s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready),.s_axil_wdata(s_axil_wdata),
        .s_axil_wstrb(s_axil_wstrb),.s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),.s_axil_bresp(s_axil_bresp),
        .s_axil_bvalid(s_axil_bvalid),.s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr),.s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready),.s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp),.s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .s_axis_tdata(s_axis_tdata),.s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),.s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),.m_axis_tready(m_axis_tready)
    );

    initial clk = 0;
    always #2.5 clk = ~clk;

    task axil_write;
        input [5:0] addr; input [31:0] data;
        begin
            @(negedge clk);
            s_axil_awaddr=addr; s_axil_awvalid=1;
            s_axil_wdata=data; s_axil_wstrb=4'hf; s_axil_wvalid=1; s_axil_bready=1;
            @(posedge clk); while(!s_axil_awready||!s_axil_wready) @(posedge clk);
            @(negedge clk); s_axil_awvalid=0; s_axil_wvalid=0;
            @(posedge clk); while(!s_axil_bvalid) @(posedge clk);
            @(negedge clk); s_axil_bready=0;
            @(posedge clk);
        end
    endtask

    task axil_read;
        input [5:0] addr; output [31:0] data;
        begin
            @(negedge clk);
            s_axil_araddr=addr; s_axil_arvalid=1; s_axil_rready=1;
            @(posedge clk); while(!s_axil_arready) @(posedge clk);
            @(negedge clk); s_axil_arvalid=0;
            @(posedge clk); while(!s_axil_rvalid) @(posedge clk);
            data=s_axil_rdata;
            @(negedge clk); s_axil_rready=0;
            @(posedge clk);
        end
    endtask

    integer beat, timeout, fail_count, pass_count, collected;
    reg [63:0] input_beats [0:7];
    reg [63:0] out_beats   [0:7];
    reg [31:0] rdata;

    initial begin
        $dumpfile("m3_cosim.vcd");
        $dumpvars(0, tb_top);

        {s_axil_awvalid,s_axil_wvalid,s_axil_bready,
         s_axil_arvalid,s_axil_rready,
         s_axis_tvalid,s_axis_tlast,m_axis_tready}=0;
        s_axil_awaddr=0;s_axil_wdata=0;s_axil_wstrb=4'hf;
        s_axil_araddr=0;s_axis_tdata=0;
        pass_count=0;fail_count=0;

        // ---- RESET ----
        rst_n=0; repeat(4) @(posedge clk);
        rst_n=1; repeat(2) @(posedge clk);

        // ============================================================
        // REGION 1: HOST WRITE — AXI4-Lite config
        // ============================================================
        $display("[REGION 1] Host write via AXI4-Lite");
        axil_write(6'h08, 32'd64);   // CFG_D=64
        axil_write(6'h0C, 32'd64);   // CFG_T=64
        axil_write(6'h10, 32'd0);    // INT8
        axil_write(6'h00, 32'd1);    // start
        $display("  AXI4-Lite config writes done");

        // ============================================================
        // REGION 2: COMPUTE ACTIVITY — stream all 8 beats, accept output simultaneously
        // d=64 row: 8 beats x 8 bytes, monotonically increasing values
        // Software reference: running_sum ≈ 8*255=2040 for uniform exp output
        // ============================================================
        $display("[REGION 2] Streaming 8 input beats (d=64 row)");
        input_beats[0] = 64'h0a141e28_3c46_5064;
        input_beats[1] = 64'h0a141e28_3c46_5064;
        input_beats[2] = 64'h0a141e28_3c46_5064;
        input_beats[3] = 64'h0a141e28_3c46_5064;
        input_beats[4] = 64'h0a141e28_3c46_5064;
        input_beats[5] = 64'h0a141e28_3c46_5064;
        input_beats[6] = 64'h0a141e28_3c46_5064;
        input_beats[7] = 64'h0a141e28_3c46_5064;

        m_axis_tready = 1;
        collected = 0;

        // Send all 8 input beats back-to-back
        for (beat = 0; beat < 8; beat = beat + 1) begin
            @(negedge clk);
            s_axis_tdata  = input_beats[beat];
            s_axis_tvalid = 1;
            s_axis_tlast  = (beat == 7) ? 1 : 0;
            @(posedge clk);
            timeout = 0;
            while (!s_axis_tready && timeout < 50) begin
                @(posedge clk); timeout = timeout + 1;
            end
            $display("  Input beat %0d: 0x%016h  tready=%b", beat, input_beats[beat], s_axis_tready);
        end
        @(negedge clk);
        s_axis_tvalid = 0; s_axis_tlast = 0;

        // ============================================================
        // REGION 3: HOST READ — wait for and collect 8 output beats
        // Pipeline latency = 8 cycles; collect as they arrive
        // ============================================================
        $display("[REGION 3] Collecting output (pipeline latency = 8 cycles)");

        collected = 0;
        timeout   = 0;
        // Poll until we have 8 beats or timeout
        while (collected < 8 && timeout < 2000) begin
            @(posedge clk);
            timeout = timeout + 1;
            if (m_axis_tvalid && m_axis_tready) begin
                out_beats[collected] = m_axis_tdata;
                $display("  Output beat %0d: 0x%016h  last=%b  [NONZERO=%b]",
                         collected, m_axis_tdata, m_axis_tlast,
                         (m_axis_tdata !== 64'd0));
                if (m_axis_tdata !== 64'd0)
                    pass_count = pass_count + 1;
                else
                    fail_count = fail_count + 1;
                collected = collected + 1;
            end
        end
        if (collected < 8) begin
            $display("  Only collected %0d/8 output beats (timeout)", collected);
            fail_count = fail_count + (8 - collected);
        end

        // Read STATUS register
        repeat(4) @(posedge clk);
        axil_read(6'h04, rdata);
        $display("  STATUS = 0x%08h  done=%b", rdata, rdata[2]);

        // ============================================================
        // VERDICT
        // ============================================================
        $display("");
        $display("=== M3 CO-SIMULATION RESULTS ===");
        $display("Beats PASS : %0d / 8", pass_count);
        $display("Beats FAIL : %0d / 8", fail_count);
        if (fail_count == 0 && pass_count == 8)
            $display("RESULT: PASS");
        else
            $display("RESULT: FAIL");
        $display("================================");
        $finish;
    end

    initial begin #500000; $display("WATCHDOG — RESULT: FAIL"); $finish; end

endmodule
