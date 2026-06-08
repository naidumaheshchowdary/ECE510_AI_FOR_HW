// =============================================================================
// interface.sv — AXI4-Lite Control + AXI4-Stream Data Interface
// ECE 410/510 HW4AI | Spring 2026 | Milestone 4
// Author : Mahesh Chowdary Naidu
// =============================================================================
// Change from M3: module renamed interface_mod_m3 → interface_mod
//                 instantiates compute_core (not compute_core_m3)
// All AXI4-Lite and AXI4-Stream logic identical to M2/M3.
// =============================================================================
//
// AXI4-Lite Register Map (6-bit address):
//   0x00 CTRL      [0]=start [1]=soft_reset
//   0x04 STATUS    [2]=done (read-only)
//   0x08 CFG_D     [7:0] row width (default 64)
//   0x0C CFG_T     [7:0] num rows (default 64)
//   0x10 PRECISION [0]=0 INT8, [0]=1 FP64
// =============================================================================
`timescale 1ns/1ps
module interface_mod #(
    parameter AXIL_AW = 6,
    parameter AXIL_DW = 32,
    parameter AXIS_W  = 64,
    parameter D       = 64,
    parameter T       = 64
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire  [AXIL_AW-1:0]   s_axil_awaddr,
    input  wire                   s_axil_awvalid,
    output reg                    s_axil_awready,
    input  wire  [AXIL_DW-1:0]   s_axil_wdata,
    input  wire  [3:0]            s_axil_wstrb,
    input  wire                   s_axil_wvalid,
    output reg                    s_axil_wready,
    output reg   [1:0]            s_axil_bresp,
    output reg                    s_axil_bvalid,
    input  wire                   s_axil_bready,
    input  wire  [AXIL_AW-1:0]   s_axil_araddr,
    input  wire                   s_axil_arvalid,
    output reg                    s_axil_arready,
    output reg   [AXIL_DW-1:0]   s_axil_rdata,
    output reg   [1:0]            s_axil_rresp,
    output reg                    s_axil_rvalid,
    input  wire                   s_axil_rready,
    input  wire  [AXIS_W-1:0]    s_axis_tdata,
    input  wire                   s_axis_tvalid,
    input  wire                   s_axis_tlast,
    output wire                   s_axis_tready,
    output wire  [AXIS_W-1:0]    m_axis_tdata,
    output wire                   m_axis_tvalid,
    output wire                   m_axis_tlast,
    input  wire                   m_axis_tready
);
    reg [AXIL_DW-1:0] reg_ctrl;
    reg [AXIL_DW-1:0] reg_status;
    reg [AXIL_DW-1:0] reg_cfg_d;
    reg [AXIL_DW-1:0] reg_cfg_t;
    reg [AXIL_DW-1:0] reg_precision;
    wire core_done;

    // AXI4-Lite write channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl<=32'd0; reg_cfg_d<=D; reg_cfg_t<=T; reg_precision<=32'd0;
            s_axil_awready<=1'b0; s_axil_wready<=1'b0;
            s_axil_bvalid<=1'b0; s_axil_bresp<=2'b00;
        end else begin
            s_axil_awready<=1'b1; s_axil_wready<=1'b1;
            if (s_axil_awvalid && s_axil_wvalid) begin
                case (s_axil_awaddr)
                    6'h00: reg_ctrl     <=s_axil_wdata;
                    6'h08: reg_cfg_d    <=s_axil_wdata;
                    6'h0C: reg_cfg_t    <=s_axil_wdata;
                    6'h10: reg_precision<=s_axil_wdata;
                    default: ;
                endcase
                s_axil_bvalid<=1'b1; s_axil_bresp<=2'b00;
            end else if (s_axil_bready) s_axil_bvalid<=1'b0;
        end
    end

    // AXI4-Lite read channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_arready<=1'b0; s_axil_rvalid<=1'b0;
            s_axil_rdata<=32'd0; s_axil_rresp<=2'b00;
        end else begin
            s_axil_arready<=1'b1;
            if (s_axil_arvalid) begin
                s_axil_rvalid<=1'b1; s_axil_rresp<=2'b00;
                case (s_axil_araddr)
                    6'h00: s_axil_rdata<=reg_ctrl;
                    6'h04: s_axil_rdata<=reg_status;
                    6'h08: s_axil_rdata<=reg_cfg_d;
                    6'h0C: s_axil_rdata<=reg_cfg_t;
                    6'h10: s_axil_rdata<=reg_precision;
                    default: s_axil_rdata<=32'd0;
                endcase
            end else if (s_axil_rready) s_axil_rvalid<=1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) reg_status<=32'd0;
        else        reg_status[2]<=core_done;
    end

    // Instantiate M4 compute core
    compute_core #(.D(D),.T(T),.DATA_W(8),.AXIS_W(AXIS_W),.PIPE_DEPTH(8)) u_core (
        .clk(clk),.rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),.s_axis_tvalid(s_axis_tvalid),
        .s_axis_tlast(s_axis_tlast),.s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),.m_axis_tvalid(m_axis_tvalid),
        .m_axis_tlast(m_axis_tlast),.m_axis_tready(m_axis_tready),
        .cfg_d(reg_cfg_d[7:0]),.cfg_t(reg_cfg_t[7:0]),
        .precision(reg_precision[0]),.start(reg_ctrl[0]),.done(core_done)
    );
endmodule
