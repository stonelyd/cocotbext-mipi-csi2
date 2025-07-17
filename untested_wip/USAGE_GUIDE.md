# CSI-2 Extension Usage Guide

This guide demonstrates how to use the cocotbext-csi2 extension for MIPI CSI-2 simulation.

## Quick Start

```python
from cocotbext.csi2 import *
from cocotbext.csi2.config import Csi2Config, PhyType, DataType
from cocotbext.csi2.packet import Csi2ShortPacket, Csi2LongPacket

# Configure D-PHY
config = Csi2Config(
    phy_type=PhyType.DPHY,
    lane_count=4,
    bit_rate_mbps=2500,
    continuous_clock=True
)

# Create test bus and models
bus = Csi2DPhyBus(dut, lane_count=4)
tx_model = Csi2TxModel(bus, config)
rx_model = Csi2RxModel(bus, config)
```

## Configuration Examples

### D-PHY Configuration
```python
dphy_config = Csi2Config(
    phy_type=PhyType.DPHY,
    lane_count=4,                    # 1, 2, 4, or 8 lanes
    bit_rate_mbps=2500,             # Data rate per lane
    continuous_clock=True,           # Clock mode
    scrambling_enabled=True,         # CSI-2 v2.0+ feature
    virtual_channel_extension=True   # Extended VC support
)
```

### C-PHY Configuration
```python
cphy_config = Csi2Config(
    phy_type=PhyType.CPHY,
    trio_count=3,                   # 1, 2, or 3 trios
    bit_rate_mbps=3000,            # Symbol rate
    lane_distribution_enabled=True  # Multi-trio distribution
)
```

## Packet Creation

### Short Packets
```python
# Frame control packets
frame_start = Csi2ShortPacket.frame_start(0, 1001)  # VC=0, Frame=1001
frame_end = Csi2ShortPacket.frame_end(0, 1001)

# Line control packets  
line_start = Csi2ShortPacket.line_start(0, 480)     # VC=0, Line=480
line_end = Csi2ShortPacket.line_end(0, 480)

# Generic short packets
generic = Csi2ShortPacket(0, DataType.GENERIC_SHORT_PACKET_1, 0x1234)
```

### Long Packets
```python
# Image data packet
image_data = generate_test_pattern(640, 480, "ramp")
raw8_packet = Csi2LongPacket(0, DataType.RAW8, image_data)

# RGB data packet
rgb_data = generate_test_pattern(320, 240, "checkerboard")
rgb_packet = Csi2LongPacket(1, DataType.RGB888, rgb_data)

# Custom payload
custom_data = bytes([i for i in range(256)])
custom_packet = Csi2LongPacket(2, DataType.USER_DEFINED_1, custom_data)
```

## Data Types

The extension supports all standard CSI-2 data types:

```python
# Raw formats
DataType.RAW8, DataType.RAW10, DataType.RAW12, DataType.RAW16
DataType.RAW6, DataType.RAW7, DataType.RAW14, DataType.RAW20

# RGB formats
DataType.RGB888, DataType.RGB666, DataType.RGB565, DataType.RGB444

# YUV formats
DataType.YUV422_8BIT, DataType.YUV420_8BIT, DataType.YUV422_10BIT

# Control packets
DataType.FRAME_START, DataType.FRAME_END
DataType.LINE_START, DataType.LINE_END

# Generic packets
DataType.GENERIC_SHORT_PACKET_1 to DataType.GENERIC_SHORT_PACKET_8
DataType.GENERIC_LONG_PACKET_1 to DataType.GENERIC_LONG_PACKET_4
```

## Virtual Channels

```python
from cocotbext.csi2.config import VirtualChannel

# Standard virtual channels (CSI-2 v1.x)
VirtualChannel.VC0, VirtualChannel.VC1, VirtualChannel.VC2, VirtualChannel.VC3

# Extended virtual channels (CSI-2 v2.0+)
VirtualChannel.VC4 through VirtualChannel.VC15
```

## Basic Simulation Test

```python
import cocotb
from cocotb.triggers import Timer
from cocotb.clock import Clock

@cocotb.test()
async def test_csi2_transmission(dut):
    # Setup clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure CSI-2
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=2,
        bit_rate_mbps=1000
    )
    
    # Create models
    bus = Csi2DPhyBus(dut, lane_count=2)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send frame
    await tx_model.send_frame(160, 120, DataType.RAW8, 0, 1)
    
    # Verify reception
    frame_received = await rx_model.wait_for_frame(0, timeout_ns=50000)
    assert frame_received, "Frame not received"
```

## Error Injection Testing

```python
# Configure error injection
error_config = Csi2Config(
    phy_type=PhyType.DPHY,
    lane_count=4,
    bit_rate_mbps=1500,
    inject_ecc_errors=True,
    inject_checksum_errors=True,
    error_injection_rate=0.1  # 10% error rate
)

# Create models with error injection
tx_model = Csi2TxModel(bus, error_config)
rx_model = Csi2RxModel(bus, error_config)

# Enable error detection callback
async def error_handler(error_type, packet, message):
    cocotb.log.info(f"Error detected: {error_type}")

rx_model.on_error_detected = error_handler
```

## Multi-Lane Performance Testing

```python
@cocotb.test()
async def test_multilane_performance(dut):
    lane_configs = [1, 2, 4, 8]
    
    for lanes in lane_configs:
        config = Csi2Config(
            phy_type=PhyType.DPHY,
            lane_count=lanes,
            bit_rate_mbps=2000,
            lane_distribution_enabled=True
        )
        
        bus = Csi2DPhyBus(dut, lane_count=lanes)
        tx_model = Csi2TxModel(bus, config)
        rx_model = Csi2RxModel(bus, config)
        
        # Measure throughput
        start_time = cocotb.utils.get_sim_time('ns')
        await tx_model.send_frame(640, 480, DataType.RAW8, 0, 0)
        await rx_model.wait_for_frame(0, timeout_ns=100000)
        end_time = cocotb.utils.get_sim_time('ns')
        
        throughput = (640 * 480 * 8) / ((end_time - start_time) / 1000)
        cocotb.log.info(f"{lanes}-lane throughput: {throughput:.1f} Mbps")
```

## Utilities

### Test Pattern Generation
```python
from cocotbext.csi2.utils import generate_test_pattern

# Available patterns
ramp_data = generate_test_pattern(640, 480, "ramp")
checker_data = generate_test_pattern(320, 240, "checkerboard") 
solid_data = generate_test_pattern(160, 120, "solid")
```

### ECC and Checksum Validation
```python
from cocotbext.csi2.utils import calculate_ecc, validate_checksum

# Calculate ECC for header
data_id = (0 << 6) | DataType.RAW8  # VC=0, DT=RAW8
ecc = calculate_ecc(data_id, 1024)

# Validate payload checksum
payload = bytes([i for i in range(100)])
checksum = sum(payload) & 0xFFFF
is_valid = validate_checksum(payload, checksum)
```

### RAW10 Packing/Unpacking
```python
from cocotbext.csi2.utils import pack_raw10, unpack_raw10

# Pack 10-bit pixels into CSI-2 RAW10 format
pixels = [i * 4 for i in range(256)]  # 10-bit values
packed_data = pack_raw10(pixels)

# Unpack RAW10 data back to pixel values
unpacked_pixels = unpack_raw10(packed_data)
```

## Bus Abstraction

### Automatic Signal Detection
```python
# Automatically detect CSI-2 signals
bus = Csi2Bus.from_entity(dut)

# Or specify prefix
bus = Csi2Bus.from_prefix(dut, "csi2_")
```

### Manual Bus Creation
```python
# D-PHY bus
dphy_bus = Csi2DPhyBus(dut, lane_count=4)

# C-PHY bus  
cphy_bus = Csi2CPhyBus(dut, trio_count=2)
```

## Advanced Features

### Continuous Streaming
```python
# Start continuous image streaming
image_stream = Csi2ImageTransmitter(tx_model, 640, 480, DataType.RAW8, 0)
await image_stream.start(frame_rate_fps=30)

# Continuous reception
image_receiver = Csi2ImageReceiver(rx_model, 0)
await image_receiver.start()
```

### Multi-Virtual Channel
```python
# Send concurrent streams on different VCs
tasks = []
for vc in range(4):
    task = cocotb.start_soon(
        tx_model.send_frame(160, 120, DataType.RAW8, vc, vc)
    )
    tasks.append(task)

# Wait for all transmissions
await asyncio.gather(*tasks)
```

### Timing Validation
```python
# Enable strict timing validation
rx_model.enable_strict_timing_validation(True)

# Check for timing violations
stats = rx_model.get_statistics()
if stats['timing_violations'] > 0:
    cocotb.log.warning(f"Detected {stats['timing_violations']} timing violations")
```

## Statistics and Monitoring

```python
# Get transmission statistics
tx_stats = tx_model.get_statistics()
print(f"Packets sent: {tx_stats['packets_sent']}")
print(f"Errors injected: {tx_stats['errors_injected']}")

# Get reception statistics  
rx_stats = rx_model.get_statistics()
print(f"Packets received: {rx_stats['packets_received']}")
print(f"ECC errors: {rx_stats['ecc_errors']}")
print(f"Checksum errors: {rx_stats['checksum_errors']}")
```

This extension provides comprehensive MIPI CSI-2 simulation capabilities for validating camera interfaces, testing protocol compliance, and analyzing performance characteristics.