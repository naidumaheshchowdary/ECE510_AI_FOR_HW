// =============================================================================
// top.sv — Integrated Top Module, Milestone 4
// ECE 410/510 HW4AI | Spring 2026
// Author : Mahesh Chowdary Naidu
// =============================================================================
// Thin wrapper instantiating interface_mod (which contains compute_core).
// No glue logic — single clock domain, same data width throughout.
// Change from M3: instantiates interface_mod (not interface_mod_m3)
// =============================================================================
`timescale 1ns/1ps
module top #(
    parameter AXIL_AW = 6,
    parameter AXIL_DW = 32,
    parameter AXIS_W  = 64,
    parameter D       = 64,
    parameter T       = 64
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire  [AXIL_AW-1:0]  s_axil_awaddr,
    input  wire                  s_axil_awvalid,
    output wire                  s_axil_awready,
    input  wire  [AXIL_DW-1:0]  s_axil_wdata,
    input  wire  [3:0]           s_axil_wstrb,
    input  wire                  s_axil_wvalid,
    output wire                  s_axil_wready,
    output wire  [1:0]           s_axil_bresp,
    output wire                  s_axil_bvalid,
    input  wire                  s_axil_bready,
    input  wire  [AXIL_AW-1:0]  s_axil_araddr,
    input  wire                  s_axil_arvalid,
    output wire                  s_axil_arready,
    output wire  [AXIL_DW-1:0]  s_axil_rdata,
    output wire  [1:0]           s_axil_rresp,
    output wire                  s_axil_rvalid,
    input  wire                  s_axil_rready,
    input  wire  [AXIS_W-1:0]   s_axis_tdata,
    input  wire                  s_axis_tvalid,
    input  wire                  s_axis_tlast,
    output wire                  s_axis_tready,
    output wire  [AXIS_W-1:0]   m_axis_tdata,
    output wire                  m_axis_tvalid,
    output wire                  m_axis_tlast,
    input  wire                  m_axis_tready
);
    interface_mod #(
        .AXIL_AW(AXIL_AW),.AXIL_DW(AXIL_DW),.AXIS_W(AXIS_W),.D(D),.T(T)
    ) u_interface (
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
endmodule
