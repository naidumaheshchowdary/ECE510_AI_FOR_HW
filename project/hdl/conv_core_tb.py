# conv_core_tb.py — cocotb testbench stub for conv_core.sv
# ECE 510 Codefest 4 COPT Part B

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def test_conv_core_reset(dut):
    """Verify accumulator clears to 0 on reset."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst.value      = 1
    dut.valid_in.value = 0
    for i in range(4):
        dut.act[i].value = 0
        dut.wgt[i].value = 0

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")

    assert dut.accum_out.value.signed_integer == 0
    dut._log.info("PASS: reset -> accum_out = 0")


@cocotb.test()
async def test_conv_core_basic(dut):
    """Apply act=[1,2,3,4] wgt=[4,3,2,1] -> lane_sum=20."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst.value      = 1
    dut.valid_in.value = 0
    for i in range(4):
        dut.act[i].value = 0
        dut.wgt[i].value = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.rst.value = 0

    for i, (a, w) in enumerate(zip([1,2,3,4], [4,3,2,1])):
        dut.act[i].value = a
        dut.wgt[i].value = w
    dut.valid_in.value = 1

    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    result = dut.accum_out.value.signed_integer
    assert result == 20, f"Expected 20, got {result}"
    dut._log.info(f"PASS: accum_out = {result}")

    # Negative weights: act=[1,1,1,1] wgt=[-1,-1,-1,-1] -> accum = 20-4 = 16
    for i in range(4):
        dut.act[i].value = 1
        dut.wgt[i].value = -1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    result = dut.accum_out.value.signed_integer
    assert result == 16, f"Expected 16, got {result}"
    dut._log.info(f"PASS: accum_out = {result}")
    dut._log.info("test_conv_core_basic: ALL PASSED")
