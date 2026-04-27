import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

CLK_NS   = 10
D        = 64
BEAT_W   = 8
BEATS    = D // BEAT_W
PIPE_D   = 8


async def reset_dut(dut):
    dut.rst_n.value          = 0
    dut.s_axis_tvalid.value  = 0
    dut.s_axis_tdata.value   = 0
    dut.s_axis_tlast.value   = 0
    dut.m_axis_tready.value  = 1
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wvalid.value  = 0
    dut.s_axil_arvalid.value = 0
    dut.s_axil_bready.value  = 1
    dut.s_axil_rready.value  = 1
    dut.s_axil_awaddr.value  = 0
    dut.s_axil_wdata.value   = 0
    dut.s_axil_wstrb.value   = 0xF
    dut.s_axil_araddr.value  = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")


async def axil_write(dut, addr, data):
    dut.s_axil_awaddr.value  = addr
    dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value   = data
    dut.s_axil_wstrb.value   = 0xF
    dut.s_axil_wvalid.value  = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wvalid.value  = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        if dut.s_axil_bvalid.value == 1:
            break


async def axil_read(dut, addr):
    dut.s_axil_araddr.value  = addr
    dut.s_axil_arvalid.value = 1
    # Wait TWO edges: one for arready, one for rvalid to appear
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.s_axil_arvalid.value = 0
    # Sample rdata — rvalid should be high now
    for _ in range(10):
        if dut.s_axil_rvalid.value == 1:
            val = int(dut.s_axil_rdata.value)
            # deassert arvalid and wait for rvalid to clear
            await RisingEdge(dut.clk)
            await Timer(1, units="ns")
            return val
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
    raise AssertionError("AXI4-Lite read timed out")


@cocotb.test()
async def test_reset(dut):
    """Verify all outputs clear after active-low reset."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset_dut(dut)
    assert dut.m_axis_tvalid.value == 0, f"m_axis_tvalid should be 0, got {dut.m_axis_tvalid.value}"
    assert dut.s_axis_tready.value == 1, f"s_axis_tready should be 1, got {dut.s_axis_tready.value}"
    dut._log.info("PASS: test_reset")


@cocotb.test()
async def test_axil_registers(dut):
    """Write and read back AXI4-Lite control registers."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset_dut(dut)

    await axil_write(dut, 0x08, 64)
    val = await axil_read(dut, 0x08)
    assert val == 64, f"CFG_D: expected 64, got {val}"
    dut._log.info(f"PASS: CFG_D = {val}")

    await axil_write(dut, 0x0C, 64)
    val = await axil_read(dut, 0x0C)
    assert val == 64, f"CFG_T: expected 64, got {val}"
    dut._log.info(f"PASS: CFG_T = {val}")

    await axil_write(dut, 0x10, 0)
    val = await axil_read(dut, 0x10)
    assert val == 0, f"PRECISION: expected 0, got {val}"
    dut._log.info(f"PASS: PRECISION = {val} (INT8)")

    await axil_write(dut, 0x00, 1)
    val = await axil_read(dut, 0x00)
    assert val == 1, f"CTRL: expected 1, got {val}"
    dut._log.info(f"PASS: CTRL start asserted")

    dut._log.info("PASS: test_axil_registers")


@cocotb.test()
async def test_axis_stream(dut):
    """Send one row (8 beats x 8 bytes = 64 INT8 elements). Verify pipeline propagates."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset_dut(dut)

    dut.m_axis_tready.value = 1

    for idx in range(BEATS):
        val = 0
        for b in range(BEAT_W):
            val |= ((idx + 1) & 0xFF) << (b * 8)
        dut.s_axis_tdata.value  = val
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value  = 1 if idx == BEATS - 1 else 0
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")

    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0

    output_seen = False
    last_seen   = False
    for cycle in range(PIPE_D + BEATS + 5):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        if dut.m_axis_tvalid.value == 1:
            output_seen = True
            dut._log.info(f"  cycle {cycle}: tdata=0x{int(dut.m_axis_tdata.value):016X} tlast={int(dut.m_axis_tlast.value)}")
            if dut.m_axis_tlast.value == 1:
                last_seen = True
                break

    assert output_seen, "No output observed from pipeline"
    assert last_seen,   "tlast never seen — row did not complete"
    dut._log.info("PASS: test_axis_stream")
