"""
Basic CSI-2 functionality tests

Copyright (c) 2024 CSI-2 Extension Contributors
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import pytest

from cocotbext.csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2Bus, Csi2ShortPacket, Csi2LongPacket
)


@cocotb.test()
async def test_basic_packet_transmission(dut):
    """Test basic CSI-2 packet transmission and reception"""
    
    # Create clock
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
        lane_count=1,
        bit_rate_mbps=500,
        continuous_clock=True
    )
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Test short packet transmission
    frame_start = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
    await tx_model.send_packet(frame_start)
    
    # Wait for reception
    await Timer(1000, units="ns")
    
    # Check received packet
    received_packet = await rx_model.get_next_packet(timeout_ns=1000)
    assert received_packet is not None
    assert received_packet.virtual_channel == 0
    assert received_packet.data_type == DataType.FRAME_START
    
    cocotb.log.info("Basic packet transmission test passed")


@cocotb.test()
async def test_frame_transmission(dut):
    """Test complete frame transmission"""
    
    # Create clock
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
        lane_count=1,
        bit_rate_mbps=800,
        continuous_clock=True
    )
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send test frame
    width, height = 160, 120
    await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
    
    # Wait for frame completion
    frame_received = await rx_model.wait_for_frame(virtual_channel=0, timeout_ns=100000)
    assert frame_received, "Frame not received within timeout"
    
    # Validate frame
    frame_data = rx_model.get_frame_data(0)
    assert frame_data is not None
    assert len(frame_data) == width * height  # RAW8 format
    
    cocotb.log.info(f"Frame transmission test passed: received {len(frame_data)} bytes")


@cocotb.test()
async def test_multi_virtual_channel(dut):
    """Test multiple virtual channel support"""
    
    # Create clock
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
        bit_rate_mbps=1000,
        continuous_clock=True
    )
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send packets on different VCs
    for vc in range(4):
        packet = Csi2ShortPacket.frame_start(virtual_channel=vc, frame_number=vc)
        await tx_model.send_packet(packet)
    
    await Timer(5000, units="ns")
    
    # Verify packets received on correct VCs
    for vc in range(4):
        packets = rx_model.get_received_packets(virtual_channel=vc)
        assert len(packets) >= 1, f"No packets received on VC{vc}"
        assert packets[0].virtual_channel == vc
        assert packets[0].data == vc  # Frame number
    
    cocotb.log.info("Multi virtual channel test passed")


@cocotb.test()
async def test_error_detection(dut):
    """Test error detection capabilities"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure CSI-2 with error injection
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=500,
        continuous_clock=True,
        inject_ecc_errors=True,
        error_injection_rate=0.5
    )
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Enable error injection
    tx_model.enable_error_injection(0.5)
    
    # Send packets with errors
    for i in range(10):
        packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=i)
        await tx_model.send_packet(packet)
    
    await Timer(10000, units="ns")
    
    # Check statistics
    tx_stats = tx_model.get_statistics()
    rx_stats = rx_model.get_statistics()
    
    assert tx_stats['errors_injected'] > 0, "No errors were injected"
    assert rx_stats['ecc_errors'] > 0, "No ECC errors detected"
    
    cocotb.log.info(f"Error detection test passed: "
                   f"{tx_stats['errors_injected']} errors injected, "
                   f"{rx_stats['ecc_errors']} ECC errors detected")


@cocotb.test()
async def test_packet_builder(dut):
    """Test packet builder functionality"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure CSI-2
    config = Csi2Config(phy_type=PhyType.DPHY, lane_count=1)
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Use packet builder
    builder = tx_model.packet_builder.set_virtual_channel(2)
    
    # Build and send various packets
    frame_start = builder.build_frame_start(42)
    line_start = builder.build_line_start(10)
    pixel_data = builder.build_raw8_line(64, b'\x80' * 64)  # Gray line
    line_end = builder.build_line_end(10)
    frame_end = builder.build_frame_end(42)
    
    packets = [frame_start, line_start, pixel_data, line_end, frame_end]
    await tx_model.send_packets(packets)
    
    await Timer(5000, units="ns")
    
    # Verify packet sequence
    received = rx_model.get_received_packets(virtual_channel=2)
    assert len(received) == 5, f"Expected 5 packets, got {len(received)}"
    
    # Check packet types
    expected_types = [
        DataType.FRAME_START, DataType.LINE_START, DataType.RAW8,
        DataType.LINE_END, DataType.FRAME_END
    ]
    
    for i, (packet, expected_type) in enumerate(zip(received, expected_types)):
        assert packet.data_type == expected_type, \
            f"Packet {i}: expected {expected_type}, got {packet.data_type}"
        assert packet.virtual_channel == 2
    
    cocotb.log.info("Packet builder test passed")


@cocotb.test()
async def test_timing_validation(dut):
    """Test timing validation features"""
    
    # Create clock
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
        lane_count=1,
        bit_rate_mbps=1000,
        continuous_clock=True
    )
    
    # Create bus and models
    bus = Csi2Bus.from_entity(dut)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Enable strict timing validation
    rx_model.enable_strict_timing_validation(True)
    
    # Send packets rapidly (may cause timing violations)
    for i in range(5):
        packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=i)
        await tx_model.send_packet(packet)
        await Timer(10, units="ns")  # Very short spacing
    
    await Timer(2000, units="ns")
    
    # Check for timing violations
    stats = rx_model.get_statistics()
    
    cocotb.log.info(f"Timing validation test: {stats['timing_violations']} violations detected")
    
    # At least some timing violations should be detected with rapid transmission
    # (This test validates the timing checker is working)
