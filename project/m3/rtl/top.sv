// =============================================================================
// top.sv — Integrated Top Module for Fused Softmax + LayerNorm Accelerator
// ECE 410/510 HW4AI | Spring 2026 | Milestone 3
// Author : Mahesh Chowdary Naidu
// =============================================================================
// Ports:
//   clk              I   1    System clock
//   rst_n            I   1    Async active-low reset
//   s_axil_awaddr    I   6    AXI4-Lite write address
//   s_axil_awvalid   I   1    AXI4-Lite write address valid
//   s_axil_awready   O   1    AXI4-Lite write address ready
//   s_axil_wdata     I  32    AXI4-Lite write data
//   s_axil_wstrb     I   4    AXI4-Lite write byte strobes
//   s_axil_wvalid    I   1    AXI4-Lite write data valid
//   s_axil_wready    O   1    AXI4-Lite write data ready
//   s_axil_bresp     O   2    AXI4-Lite write response
//   s_axil_bvalid    O   1    AXI4-Lite write response valid
//   s_axil_bready    I   1    AXI4-Lite write response ready
//   s_axil_araddr    I   6    AXI4-Lite read address
//   s_axil_arvalid   I   1    AXI4-Lite read address valid
//   s_axil_arready   O   1    AXI4-Lite read address ready
//   s_axil_rdata     O  32    AXI4-Lite read data
//   s_axil_rresp     O   2    AXI4-Lite read response
//   s_axil_rvalid    O   1    AXI4-Lite read data valid
//   s_axil_rready    I   1    AXI4-Lite read data ready
//   s_axis_tdata     I  64    AXI4-Stream input (8 x INT8 per beat)
//   s_axis_tvalid    I   1    AXI4-Stream input valid
//   s_axis_tlast     I   1    AXI4-Stream input last beat
//   s_axis_tready    O   1    AXI4-Stream input ready
//   m_axis_tdata     O  64    AXI4-Stream output (8 x INT8 per beat)
//   m_axis_tvalid    O   1    AXI4-Stream output valid
//   m_axis_tlast     O   1    AXI4-Stream output last beat
//   m_axis_tready    I   1    AXI4-Stream output ready
// =============================================================================
// Glue logic: NONE. interface_mod already instantiates compute_core internally.
// top.sv is a thin wrapper exposing interface_mod ports to the chip boundary.
// No FIFOs, no CDC, no width converters needed — single clock domain.
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
    // AXI4-Lite control
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
    // AXI4-Stream data
    input  wire  [AXIS_W-1:0]   s_axis_tdata,
    input  wire                  s_axis_tvalid,
    input  wire                  s_axis_tlast,
    output wire                  s_axis_tready,
    output wire  [AXIS_W-1:0]   m_axis_tdata,
    output wire                  m_axis_tvalid,
    output wire                  m_axis_tlast,
    input  wire                  m_axis_tready
);

    // -------------------------------------------------------------------------
    // interface_mod contains compute_core — no separate instantiation needed
    // -------------------------------------------------------------------------
    interface_mod #(
        .AXIL_AW (AXIL_AW),
        .AXIL_DW (AXIL_DW),
        .AXIS_W  (AXIS_W),
        .D       (D),
        .T       (T)
    ) u_interface (
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

endmodule
