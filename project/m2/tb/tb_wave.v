// tb_wave.v — waveform generation for GTKWave
// Run: iverilog -g2012 -o sim_wave tb/tb_wave.v rtl/compute_core.sv
//      vvp sim_wave
//      gtkwave sim/waveform.vcd
`timescale 1ns/1ps
module tb_wave;
    reg clk=0, rst_n=0;
    reg [63:0] s_axis_tdata=0;
    reg s_axis_tvalid=0, s_axis_tlast=0;
    wire s_axis_tready, m_axis_tvalid, m_axis_tlast, done;
    wire [63:0] m_axis_tdata;
    reg m_axis_tready=1;

    compute_core #(.D(64),.T(64),.PIPE_DEPTH(8)) dut (
        .clk(clk),.rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),.s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),.s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),.m_axis_tready(m_axis_tready),
        .cfg_d(8'd64),.cfg_t(8'd64),.precision(1'b0),.start(1'b1),.done(done)
    );

    always #5 clk = ~clk;

    integer beat, b;
    reg [63:0] bval;

    initial begin
        $dumpfile("sim/waveform.vcd");
        $dumpvars(0, tb_wave);
        // Reset
        rst_n=0; repeat(4) @(posedge clk); #1;
        rst_n=1; repeat(2) @(posedge clk); #1;
        // Drive 8 beats ramp 1..8
        for (beat=0; beat<8; beat=beat+1) begin
            bval=64'd0;
            for (b=0; b<8; b=b+1) bval[b*8+:8]=(beat+1)&8'hFF;
            s_axis_tdata=bval; s_axis_tvalid=1'b1;
            s_axis_tlast=(beat==7)?1'b1:1'b0;
            @(posedge clk); #1;
        end
        s_axis_tvalid=1'b0; s_axis_tlast=1'b0;
        // Wait for pipeline to drain + done
        repeat(20) @(posedge clk); #1;
        $display("waveform.vcd written — open with: gtkwave sim/waveform.vcd");
        $finish;
    end
endmodule
