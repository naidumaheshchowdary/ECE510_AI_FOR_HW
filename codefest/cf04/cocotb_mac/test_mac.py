import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


@cocotb.test()
async def test_mac_basic(dut):
    """Basic MAC: a=3,b=4 x3 cycles, reset, a=-5,b=2 x2 cycles."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Reset for one full cycle
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")   # small settle delay after edge
    dut.rst.value = 0

    # Apply inputs BEFORE the edge, sample AFTER
    dut.a.value = 3
    dut.b.value = 4
    for expected in [12, 24, 36]:
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")   # wait 1ns after edge for FF output to settle
        got = dut.out.value.signed_integer
        assert got == expected, f"Expected {expected}, got {got}"
        dut._log.info(f"PASS: out = {got}")

    # Assert reset
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    got = dut.out.value.signed_integer
    assert got == 0, f"Reset failed: got {got}"
    dut._log.info("PASS: reset -> out = 0")
    dut.rst.value = 0

    # Negative inputs
    dut.a.value = -5
    dut.b.value = 2
    for expected in [-10, -20]:
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        got = dut.out.value.signed_integer
        assert got == expected, f"Expected {expected}, got {got}"
        dut._log.info(f"PASS: out = {got}")

    dut._log.info("test_mac_basic: ALL PASSED")


@cocotb.test()
async def test_mac_overflow(dut):
    """Overflow test: accumulate until 32-bit signed wraps. Documents wrap behavior."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    dut.rst.value = 0

    # 127*127 = 16129 per cycle -> wraps at cycle ~133155
    dut.a.value = 127
    dut.b.value = 127

    prev       = 0
    wrapped    = False
    wrap_cycle = 0

    for i in range(200000):
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        cur = dut.out.value.signed_integer
        if cur < prev:
            wrapped    = True
            wrap_cycle = i + 1
            dut._log.warning(
                f"OVERFLOW at cycle {wrap_cycle}: "
                f"prev={prev}, cur={cur} -- WRAP (two's complement, no saturation)"
            )
            break
        prev = cur

    if wrapped:
        dut._log.info("RESULT: accumulator WRAPS (two's complement rollover). No saturation.")
    else:
        dut._log.info("RESULT: No overflow observed in 200000 cycles.")
