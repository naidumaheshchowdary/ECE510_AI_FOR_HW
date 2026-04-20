
# Interface Selection
## ECE 410/510 HW4AI | Spring 2026 | M1-4
## Project: Fused Softmax + Layer Normalization Accelerator

---

## 1. Host Platform

| Item | Value |
|------|-------|
| Host processor | ARM Cortex-A (SoC) |
| Host OS | Linux (embedded) |
| Host role | Runs transformer inference loop, sends activation rows to chiplet |
| Chiplet role | Computes fused softmax + layernorm, returns normalized rows |

---

## 2. Selected Interface

**Control plane:** AXI4-Lite (32-bit, memory-mapped registers)
**Data plane:** AXI4-Stream (64-bit wide, both directions)

Both are from the AMBA standard interface list specified in the project
requirements.

---

## 3. Bandwidth Requirement Calculation

One call to fused softmax + layernorm processes one T×d matrix:

```
Input data  = T × d × bytes_per_element
            = 64 × 64 × 1 byte (INT8)
            = 4,096 bytes

Output data = T × d × bytes_per_element
            = 64 × 64 × 1 byte (INT8)
            = 4,096 bytes

Total per call = 8,192 bytes
```

Target latency per call = 10 µs (4× speedup goal over 42 µs baseline).

Required bandwidth:
```
BW_required = total_bytes / target_latency
            = 8,192 bytes / 10 µs
            = 819.2 MB/s
```

---

## 4. Interface Bandwidth vs Requirement

| Interface | Bus Width | Clock | Rated BW | Required BW | Margin |
|-----------|-----------|-------|----------|-------------|--------|
| SPI | 1-bit | 50 MHz | 6.25 MB/s | 819 MB/s | ✗ 130× too slow |
| I2C | 1-bit | 1 MHz | 0.125 MB/s | 819 MB/s | ✗ 6500× too slow |
| **AXI4-Stream 64-bit** | **64-bit** | **100 MHz** | **800 MB/s** | **819 MB/s** | ≈ balanced |
| AXI4-Stream 128-bit | 128-bit | 100 MHz | 1,600 MB/s | 819 MB/s | ✓ 1.95× margin |
| PCIe Gen3 ×1 | serial | — | 985 MB/s | 819 MB/s | ✓ but overkill |
| UCIe | serial | — | >10 GB/s | 819 MB/s | ✓ but overkill |

**Selected: AXI4-Stream 64-bit at 100 MHz = 800 MB/s.**

The 64-bit AXI4-Stream provides 800 MB/s, which is approximately
balanced with the compute throughput. This means the chiplet is
interface-bound at the selected data size — a known limitation
documented here and discussed in M4.

Mitigation path for M3/M4: widen to 128-bit AXI4-Stream (1,600 MB/s)
to make the design compute-bound end-to-end.

---

## 5. AXI4-Lite Control Register Map

The control plane uses AXI4-Lite with a 6-bit address space:

| Address | Register | Description |
|---------|----------|-------------|
| 0x00 | CTRL | [0]=start, [1]=soft_reset |
| 0x04 | STATUS | [1:0]=status, [2]=done (read-only) |
| 0x08 | CFG_D | cfg_d[7:0] — row width (default 64) |
| 0x0C | CFG_T | cfg_T[7:0] — number of rows (default 64) |
| 0x10 | PRECISION | [0]=0 for INT8, 1 for FP64 |

The host writes CFG_D and CFG_T once at initialization, then writes
CTRL[0]=1 to start each inference call. It polls STATUS[2] (done) to
detect completion.

---

## 6. Interface Justification Summary

AXI4-Lite is the standard ARM AMBA control interface — it requires
only 5 channels and minimal logic, keeping the chiplet area small.
AXI4-Stream is the standard high-throughput data interface — it
provides back-pressure signaling (tready/tvalid) so the host and
chiplet can stall each other without data loss. Both are directly
supported by OpenLane 2 synthesis flow and are standard in
industry designs at Qualcomm, ARM, Intel, and NVIDIA.

SPI and I2C were rejected because they cannot sustain the required
bandwidth. PCIe and UCIe were rejected because they are designed for
board-to-board or chiplet-to-chiplet connections at much higher
bandwidth than needed, adding unnecessary complexity.
