// =============================================================================
// interface.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Author  : Mahesh Chowdary Naidu
// Project : Fused Softmax + Layer Normalization Accelerator
//
// Description:
//   Top-level interface module. Implements AXI4-Lite control register bank
//   and AXI4-Stream data path, and instantiates compute_core.
//
// NOTE: Module is named 'interface_mod' because 'interface' is a reserved
//       keyword in SystemVerilog (IEEE 1800-2012). Testbench uses interface_mod.
//
// Protocols (selected in M1 interface_selection.md — unchanged):
//   Control : AXI4-Lite  — 32-bit data, 6-bit address, 5-channel handshake
//             AWVALID/AWREADY, WVALID/WREADY, BVALID/BREADY,
//             ARVALID/ARREADY, RVALID/RREADY — all honored
//   Data    : AXI4-Stream — 64-bit TDATA (8 x INT8 per beat)
//             TVALID/TREADY back-pressure contract honored
//             TLAST marks last beat of each T×d row
//
// Clock domain : Single clock (clk)
// Reset        : Asynchronous active-low (rst_n)
//
// AXI4-Lite Register Map (6-bit address, 32-bit data):
//   Address | Register  | Description
//   --------|-----------|--------------------------------------------------
//   0x00    | CTRL      | [0]=start  [1]=soft_reset  (R/W)
//   0x04    | STATUS    | [2]=done   (read-only, set by compute_core done)
//   0x08    | CFG_D     | [7:0] row width in elements  (default 64)
//   0x0C    | CFG_T     | [7:0] number of rows          (default 64)
//   0x10    | PRECISION | [0]=0→INT8  [0]=1→FP64        (default 0)
//
// Port list (name | direction | width | purpose):
//   clk              I   1    System clock — single clock domain
//   rst_n            I   1    Asynchronous active-low reset
//   s_axil_awaddr    I   6    AXI4-Lite write address
//   s_axil_awvalid   I   1    Write address valid
//   s_axil_awready   O   1    Write address ready
//   s_axil_wdata     I  32    Write data
//   s_axil_wstrb     I   4    Write byte strobes
//   s_axil_wvalid    I   1    Write data valid
//   s_axil_wready    O   1    Write data ready
//   s_axil_bresp     O   2    Write response code (2'b00 = OKAY)
//   s_axil_bvalid    O   1    Write response valid
//   s_axil_bready    I   1    Write response ready
//   s_axil_araddr    I   6    Read address
//   s_axil_arvalid   I   1    Read address valid
//   s_axil_arready   O   1    Read address ready
//   s_axil_rdata     O  32    Read data
//   s_axil_rresp     O   2    Read response code (2'b00 = OKAY)
//   s_axil_rvalid    O   1    Read data valid
//   s_axil_rready    I   1    Read data ready
//   s_axis_tdata     I  64    AXI4-Stream input — 8 x INT8 activation elements
//   s_axis_tvalid    I   1    AXI4-Stream input valid
//   s_axis_tlast     I   1    AXI4-Stream last beat of row
//   s_axis_tready    O   1    AXI4-Stream back-pressure (passes through to core)
//   m_axis_tdata     O  64    AXI4-Stream output — 8 x INT8 normalized elements
//   m_axis_tvalid    O   1    AXI4-Stream output valid
//   m_axis_tlast     O   1    AXI4-Stream last beat of output row
//   m_axis_tready    I   1    AXI4-Stream downstream ready
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

    // AXI4-Lite slave — control
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

    // AXI4-Stream slave — input activations
    input  wire  [AXIS_W-1:0]    s_axis_tdata,
    input  wire                   s_axis_tvalid,
    input  wire                   s_axis_tlast,
    output wire                   s_axis_tready,

    // AXI4-Stream master — normalized output
    output wire  [AXIS_W-1:0]    m_axis_tdata,
    output wire                   m_axis_tvalid,
    output wire                   m_axis_tlast,
    input  wire                   m_axis_tready
);

    // -------------------------------------------------------------------------
    // Register file
    // -------------------------------------------------------------------------
    reg [AXIL_DW-1:0] reg_ctrl;       // 0x00
    reg [AXIL_DW-1:0] reg_status;     // 0x04 (RO)
    reg [AXIL_DW-1:0] reg_cfg_d;      // 0x08
    reg [AXIL_DW-1:0] reg_cfg_t;      // 0x0C
    reg [AXIL_DW-1:0] reg_precision;  // 0x10
    wire core_done;

    // -------------------------------------------------------------------------
    // AXI4-Lite write channel
    // AWVALID + WVALID sampled together (single-beat write)
    // BVALID asserted on successful write; cleared when BREADY received
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl <= 32'd0; reg_cfg_d <= D; reg_cfg_t <= T; reg_precision <= 32'd0;
            s_axil_awready <= 1'b0; s_axil_wready <= 1'b0;
            s_axil_bvalid  <= 1'b0; s_axil_bresp  <= 2'b00;
        end else begin
            s_axil_awready <= 1'b1;
            s_axil_wready  <= 1'b1;
            if (s_axil_awvalid && s_axil_wvalid) begin
                case (s_axil_awaddr)
                    6'h00: reg_ctrl      <= s_axil_wdata;
                    6'h08: reg_cfg_d     <= s_axil_wdata;
                    6'h0C: reg_cfg_t     <= s_axil_wdata;
                    6'h10: reg_precision <= s_axil_wdata;
                    default: ;
                endcase
                s_axil_bvalid <= 1'b1;
                s_axil_bresp  <= 2'b00;  // OKAY
            end else if (s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // AXI4-Lite read channel
    // RVALID asserted same cycle ARVALID seen; cleared when RREADY received
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_arready <= 1'b0; s_axil_rvalid <= 1'b0;
            s_axil_rdata   <= 32'd0; s_axil_rresp  <= 2'b00;
        end else begin
            s_axil_arready <= 1'b1;
            if (s_axil_arvalid) begin
                s_axil_rvalid <= 1'b1;
                s_axil_rresp  <= 2'b00;
                case (s_axil_araddr)
                    6'h00: s_axil_rdata <= reg_ctrl;
                    6'h04: s_axil_rdata <= reg_status;
                    6'h08: s_axil_rdata <= reg_cfg_d;
                    6'h0C: s_axil_rdata <= reg_cfg_t;
                    6'h10: s_axil_rdata <= reg_precision;
                    default: s_axil_rdata <= 32'd0;
                endcase
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    // STATUS[2] = done (set by compute_core, read via AXI4-Lite)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) reg_status <= 32'd0;
        else        reg_status[2] <= core_done;
    end

    // -------------------------------------------------------------------------
    // Compute core instantiation
    // -------------------------------------------------------------------------
    compute_core #(
        .D(D), .T(T), .DATA_W(8), .AXIS_W(AXIS_W), .PIPE_DEPTH(8)
    ) u_core (
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
        .cfg_d         (reg_cfg_d[7:0]),
        .cfg_t         (reg_cfg_t[7:0]),
        .precision     (reg_precision[0]),
        .start         (reg_ctrl[0]),
        .done          (core_done)
    );

endmodule
