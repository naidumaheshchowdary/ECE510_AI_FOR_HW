// =============================================================================
// interface.sv
// ECE 410/510 HW4AI | Spring 2026 | Project Milestone 2
// Author  : Mahesh Chowdary Naidu
// Project : Fused Softmax + Layer Normalization Accelerator
//
// Description:
//   Top-level interface module. Wraps the AXI4-Lite control register bank
//   and the AXI4-Stream data path, and connects them to the compute_core.
//
// Protocol selections (from M1 interface_selection.md):
//   Control : AXI4-Lite  — 32-bit data, 6-bit address, 5-channel handshake
//   Data    : AXI4-Stream — 64-bit TDATA, TVALID/TREADY/TLAST
//
// Clock domain : Single clock (clk)
// Reset        : Asynchronous active-low (rst_n)
//
// AXI4-Lite Register Map (6-bit address space):
//   0x00  CTRL      [0]=start  [1]=soft_reset
//   0x04  STATUS    [2]=done   (read-only, written by compute_core)
//   0x08  CFG_D     [7:0]=row width  (default 64)
//   0x0C  CFG_T     [7:0]=num rows   (default 64)
//   0x10  PRECISION [0]=0→INT8  1→FP64
//
// Port list:
//   clk              I  1    System clock
//   rst_n            I  1    Async active-low reset
//   -- AXI4-Lite slave (5 channels) --
//   s_axil_awaddr    I  6    Write address
//   s_axil_awvalid   I  1    Write address valid
//   s_axil_awready   O  1    Write address ready
//   s_axil_wdata     I  32   Write data
//   s_axil_wstrb     I  4    Write byte strobes
//   s_axil_wvalid    I  1    Write data valid
//   s_axil_wready    O  1    Write data ready
//   s_axil_bresp     O  2    Write response (00=OKAY)
//   s_axil_bvalid    O  1    Write response valid
//   s_axil_bready    I  1    Write response ready
//   s_axil_araddr    I  6    Read address
//   s_axil_arvalid   I  1    Read address valid
//   s_axil_arready   O  1    Read address ready
//   s_axil_rdata     O  32   Read data
//   s_axil_rresp     O  2    Read response (00=OKAY)
//   s_axil_rvalid    O  1    Read data valid
//   s_axil_rready    I  1    Read data ready
//   -- AXI4-Stream slave (input) --
//   s_axis_tdata     I  64   Input activation beat
//   s_axis_tvalid    I  1    Input valid
//   s_axis_tlast     I  1    Last beat of row
//   s_axis_tready    O  1    Interface ready
//   -- AXI4-Stream master (output) --
//   m_axis_tdata     O  64   Output beat
//   m_axis_tvalid    O  1    Output valid
//   m_axis_tlast     O  1    Last beat of output row
//   m_axis_tready    I  1    Downstream ready
// =============================================================================

module interface #(
    parameter int AXIL_AW = 6,
    parameter int AXIL_DW = 32,
    parameter int AXIS_W  = 64,
    parameter int D       = 64,
    parameter int T       = 64
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // AXI4-Lite slave — control
    input  logic [AXIL_AW-1:0]   s_axil_awaddr,
    input  logic                  s_axil_awvalid,
    output logic                  s_axil_awready,
    input  logic [AXIL_DW-1:0]   s_axil_wdata,
    input  logic [3:0]            s_axil_wstrb,
    input  logic                  s_axil_wvalid,
    output logic                  s_axil_wready,
    output logic [1:0]            s_axil_bresp,
    output logic                  s_axil_bvalid,
    input  logic                  s_axil_bready,
    input  logic [AXIL_AW-1:0]   s_axil_araddr,
    input  logic                  s_axil_arvalid,
    output logic                  s_axil_arready,
    output logic [AXIL_DW-1:0]   s_axil_rdata,
    output logic [1:0]            s_axil_rresp,
    output logic                  s_axil_rvalid,
    input  logic                  s_axil_rready,

    // AXI4-Stream slave — input activations
    input  logic [AXIS_W-1:0]    s_axis_tdata,
    input  logic                  s_axis_tvalid,
    input  logic                  s_axis_tlast,
    output logic                  s_axis_tready,

    // AXI4-Stream master — normalized output
    output logic [AXIS_W-1:0]    m_axis_tdata,
    output logic                  m_axis_tvalid,
    output logic                  m_axis_tlast,
    input  logic                  m_axis_tready
);

    // -------------------------------------------------------------------------
    // AXI4-Lite register bank
    // -------------------------------------------------------------------------
    logic [AXIL_DW-1:0] reg_ctrl;       // 0x00
    logic [AXIL_DW-1:0] reg_status;     // 0x04 (RO)
    logic [AXIL_DW-1:0] reg_cfg_d;      // 0x08
    logic [AXIL_DW-1:0] reg_cfg_t;      // 0x0C
    logic [AXIL_DW-1:0] reg_precision;  // 0x10

    // Control signals to compute_core
    logic core_start;
    logic core_done;

    assign core_start = reg_ctrl[0];

    // AXI4-Lite write channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_ctrl       <= '0;
            reg_cfg_d      <= AXIL_DW'(D);
            reg_cfg_t      <= AXIL_DW'(T);
            reg_precision  <= '0;
            s_axil_awready <= 1'b0;
            s_axil_wready  <= 1'b0;
            s_axil_bvalid  <= 1'b0;
            s_axil_bresp   <= 2'b00;
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

    // AXI4-Lite read channel
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axil_arready <= 1'b0;
            s_axil_rvalid  <= 1'b0;
            s_axil_rdata   <= '0;
            s_axil_rresp   <= 2'b00;
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
                    default: s_axil_rdata <= '0;
                endcase
            end else if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    // STATUS register — done bit written by compute_core
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_status <= '0;
        else
            reg_status[2] <= core_done;
    end

    // -------------------------------------------------------------------------
    // Compute core instantiation
    // -------------------------------------------------------------------------
    compute_core #(
        .D         (D),
        .T         (T),
        .DATA_W    (8),
        .AXIS_W    (AXIS_W),
        .PIPE_DEPTH(8)
    ) u_core (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tlast   (s_axis_tlast),
        .s_axis_tready  (s_axis_tready),
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tlast   (m_axis_tlast),
        .m_axis_tready  (m_axis_tready),
        .cfg_d          (reg_cfg_d[7:0]),
        .cfg_t          (reg_cfg_t[7:0]),
        .precision      (reg_precision[0]),
        .start          (core_start),
        .done           (core_done)
    );

endmodule
