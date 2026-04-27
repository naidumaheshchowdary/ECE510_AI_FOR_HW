// fused_norm_core.sv — Project Compute Core (Stub)
// ECE 510 — Hardware for AI, Spring 2026
// Author : Mahesh Chowdary Naidu
// Project: Fused Softmax + Layer Normalization Accelerator
//
// 8-stage pipeline, AXI4-Stream 64-bit data, AXI4-Lite control
// Precision: INT8 symmetric per-tensor quantization
// Target: d=64, T=64 (professor transformer config)

module fused_norm_core #(
    parameter int D          = 64,
    parameter int T          = 64,
    parameter int DATA_W     = 8,
    parameter int AXIS_W     = 64,
    parameter int AXIL_AW    = 6,
    parameter int AXIL_DW    = 32,
    parameter int PIPE_DEPTH = 8
) (
    input  logic                  clk,
    input  logic                  rst_n,

    // AXI4-Stream slave (input activations)
    input  logic [AXIS_W-1:0]     s_axis_tdata,
    input  logic                  s_axis_tvalid,
    input  logic                  s_axis_tlast,
    output logic                  s_axis_tready,

    // AXI4-Stream master (normalized output)
    output logic [AXIS_W-1:0]     m_axis_tdata,
    output logic                  m_axis_tvalid,
    output logic                  m_axis_tlast,
    input  logic                  m_axis_tready,

    // AXI4-Lite slave (control registers)
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
    input  logic                  s_axil_rready
);

    // AXI4-Lite registers
    // 0x00 CTRL     [0]=start [1]=soft_reset
    // 0x04 STATUS   [2]=done (RO)
    // 0x08 CFG_D    row width
    // 0x0C CFG_T    num rows
    // 0x10 PRECISION [0]=0->INT8 1->FP64
    logic [AXIL_DW-1:0] reg_ctrl, reg_status, reg_cfg_d, reg_cfg_t, reg_precision;

    // AXI4-Lite write
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
                s_axil_bresp  <= 2'b00;
            end else if (s_axil_bready) begin
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // AXI4-Lite read
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

    // 8-stage pipeline registers
    logic [AXIS_W-1:0] pipe_data  [PIPE_DEPTH-1:0];
    logic              pipe_valid [PIPE_DEPTH-1:0];
    logic              pipe_last  [PIPE_DEPTH-1:0];

    // Stage S1: input latch
    assign s_axis_tready = m_axis_tready | ~pipe_valid[0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe_data[0]  <= '0;
            pipe_valid[0] <= 1'b0;
            pipe_last[0]  <= 1'b0;
        end else if (s_axis_tvalid && s_axis_tready) begin
            pipe_data[0]  <= s_axis_tdata;
            pipe_valid[0] <= 1'b1;
            pipe_last[0]  <= s_axis_tlast;
        end else begin
            pipe_valid[0] <= 1'b0;
        end
    end

    // Stages S2-S8: propagate (arithmetic added in M2)
    genvar i;
    generate
        for (i = 1; i < PIPE_DEPTH; i++) begin : pipe_stage
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    pipe_data[i]  <= '0;
                    pipe_valid[i] <= 1'b0;
                    pipe_last[i]  <= 1'b0;
                end else begin
                    pipe_data[i]  <= pipe_data[i-1];
                    pipe_valid[i] <= pipe_valid[i-1];
                    pipe_last[i]  <= pipe_last[i-1];
                end
            end
        end
    endgenerate

    // Output
    assign m_axis_tdata  = pipe_data[PIPE_DEPTH-1];
    assign m_axis_tvalid = pipe_valid[PIPE_DEPTH-1];
    assign m_axis_tlast  = pipe_last[PIPE_DEPTH-1];

    // Status: done when tlast exits pipeline
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            reg_status <= '0;
        else
            reg_status[2] <= pipe_valid[PIPE_DEPTH-1] && pipe_last[PIPE_DEPTH-1];
    end

endmodule
