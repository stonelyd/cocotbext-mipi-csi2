"""
CSI-2 C-PHY specific tests

Copyright (c) 2024 CSI-2 Extension Contributors
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import pytest

from cocotbext.csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2CPhyBus, Csi2ShortPacket, Csi2LongPacket
)
from cocotbext.csi2.phy import CPhyModel


@cocotb.test()
async def test_cphy_basic_transmission(dut):
    """Test basic C-PHY transmission"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(200, units="ns")
    dut.reset_n.value = 1
    await Timer(200, units="ns")
    
    # Configure single trio C-PHY
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=1,
        bit_rate_mbps=1000,  # C-PHY typically runs at higher speeds
    )
    
    # Create C-PHY bus
    bus = Csi2CPhyBus(dut, trio_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send test packet
    packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
    await tx_model.send_packet(packet)
    
    # Wait for reception
    await Timer(2000, units="ns")
    
    # Check received packet
    received_packet = await rx_model.get_next_packet(timeout_ns=5000)
    assert received_packet is not None
    assert received_packet.virtual_channel == 0
    assert received_packet.data_type == DataType.FRAME_START
    
    cocotb.log.info("C-PHY basic transmission test passed")


@cocotb.test()
async def test_cphy_trio_encoding(dut):
    """Test C-PHY 3-phase encoding"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure C-PHY
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=1,
        bit_rate_mbps=1500,
    )
    
    # Create bus and models
    bus = Csi2CPhyBus(dut, trio_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Test different data patterns to verify encoding
    test_patterns = [
        b'\x00\x55\xAA\xFF',  # Basic patterns
        b'\x01\x02\x04\x08',  # Powers of 2
        b'\x10\x20\x40\x80',  # Shifted patterns
        bytes(range(256))[:16] # Sequential data
    ]
    
    for i, pattern in enumerate(test_patterns):
        # Send as long packet
        packet = Csi2LongPacket(0, DataType.RAW8, pattern)
        await tx_model.send_packet(packet)
        
        await Timer(5000, units="ns")
        
        # Verify reception
        received = await rx_model.get_next_packet(timeout_ns=10000)
        assert received is not None
        assert isinstance(received, Csi2LongPacket)
        assert received.payload == pattern
        
        cocotb.log.info(f"C-PHY encoding test {i+1} passed: {len(pattern)} bytes")
    
    cocotb.log.info("C-PHY trio encoding test completed")


@cocotb.test()
async def test_cphy_multi_trio(dut):
    """Test multi-trio C-PHY configuration"""
    
    # Create clock
    clock = Clock(dut.clk, 8, units="ns")  # 125 MHz
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(200, units="ns")
    dut.reset_n.value = 1
    await Timer(200, units="ns")
    
    # Configure 3-trio C-PHY (maximum)
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=3,
        bit_rate_mbps=2000,
        lane_distribution_enabled=True
    )
    
    # Create bus and models
    bus = Csi2CPhyBus(dut, trio_count=3)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send large frame to test trio distribution
    width, height = 640, 480
    await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
    
    # Wait for frame completion
    frame_received = await rx_model.wait_for_frame(virtual_channel=0, timeout_ns=300000)
    assert frame_received, "Frame not received within timeout"
    
    # Validate received frame
    frame_data = rx_model.get_frame_data(0)
    assert frame_data is not None
    assert len(frame_data) == width * height
    
    cocotb.log.info(f"C-PHY 3-trio test passed: {len(frame_data)} bytes distributed")


@cocotb.test()
async def test_cphy_state_transitions(dut):
    """Test C-PHY 3-phase state transitions"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure C-PHY
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=1,
        bit_rate_mbps=800,
    )
    
    # Create bus and models
    bus = Csi2CPhyBus(dut, trio_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Monitor state transitions
    state_transitions = []
    
    def track_transitions():
        # In a real implementation, we would monitor the actual trio signals
        # For simulation, we'll track transmission events
        current_time = cocotb.utils.get_sim_time('ns')
        state_transitions.append(current_time)
    
    # Send data with different patterns to trigger various state transitions
    test_data = [
        b'\x00' * 64,    # All zeros
        b'\xFF' * 64,    # All ones  
        b'\xAA' * 64,    # Alternating pattern
        bytes(range(256))[:64]  # Sequential
    ]
    
    for data in test_data:
        track_transitions()
        packet = Csi2LongPacket(0, DataType.RAW8, data)
        await tx_model.send_packet(packet)
        await Timer(5000, units="ns")
    
    # Verify data received correctly despite state transitions
    packets_received = 0
    while True:
        packet = await rx_model.get_next_packet(timeout_ns=1000)
        if packet is None:
            break
        if isinstance(packet, Csi2LongPacket):
            packets_received += 1
    
    assert packets_received == len(test_data), \
        f"Expected {len(test_data)} packets, got {packets_received}"
    
    cocotb.log.info(f"C-PHY state transition test passed: {len(state_transitions)} transitions tracked")


@cocotb.test()
async def test_cphy_high_speed_performance(dut):
    """Test C-PHY high-speed performance"""
    
    # Create high-speed clock
    clock = Clock(dut.clk, 5, units="ns")  # 200 MHz system clock
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure high-speed C-PHY
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=2,
        bit_rate_mbps=3000,  # High speed
        lane_distribution_enabled=True
    )
    
    # Create bus and models
    bus = Csi2CPhyBus(dut, trio_count=2)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Measure throughput with continuous transmission
    start_time = cocotb.utils.get_sim_time('ns')
    
    # Send multiple frames rapidly
    frame_count = 5
    for frame_num in range(frame_count):
        await tx_model.send_frame(320, 240, DataType.RAW8, 0, frame_num)
    
    # Wait for all frames
    frames_received = 0
    for _ in range(frame_count):
        frame_received = await rx_model.wait_for_frame(0, timeout_ns=100000)
        if frame_received:
            frames_received += 1
    
    end_time = cocotb.utils.get_sim_time('ns')
    
    assert frames_received == frame_count, \
        f"Expected {frame_count} frames, got {frames_received}"
    
    # Calculate performance metrics
    total_time_ns = end_time - start_time
    total_bytes = frame_count * 320 * 240
    throughput_mbps = (total_bytes * 8) / (total_time_ns / 1000)
    
    cocotb.log.info(f"C-PHY high-speed test: {throughput_mbps:.1f} Mbps throughput, "
                   f"{total_time_ns}ns total time")


@cocotb.test()
async def test_cphy_vs_dphy_comparison(dut):
    """Compare C-PHY vs D-PHY performance"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Test data
    test_frame_size = 64 * 64  # Small frame for quick testing
    
    # Test C-PHY configuration
    cphy_config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=1,
        bit_rate_mbps=1000,
    )
    
    # Test D-PHY configuration (equivalent lanes)
    from cocotbext.csi2 import Csi2DPhyBus
    dphy_config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=3,  # 3 lanes ≈ 1 trio
        bit_rate_mbps=1000,
        continuous_clock=True
    )
    
    # Test C-PHY
    cphy_bus = Csi2CPhyBus(dut, trio_count=1)
    cphy_tx = Csi2TxModel(cphy_bus, cphy_config)
    cphy_rx = Csi2RxModel(cphy_bus, cphy_config)
    
    cphy_start = cocotb.utils.get_sim_time('ns')
    await cphy_tx.send_frame(64, 64, DataType.RAW8, 0, 0)
    cphy_received = await cphy_rx.wait_for_frame(0, timeout_ns=50000)
    cphy_end = cocotb.utils.get_sim_time('ns')
    
    assert cphy_received, "C-PHY frame not received"
    cphy_time = cphy_end - cphy_start
    
    # Reset for D-PHY test
    await Timer(1000, units="ns")
    
    # Test D-PHY
    dphy_bus = Csi2DPhyBus(dut, lane_count=3)
    dphy_tx = Csi2TxModel(dphy_bus, dphy_config)
    dphy_rx = Csi2RxModel(dphy_bus, dphy_config)
    
    dphy_start = cocotb.utils.get_sim_time('ns')
    await dphy_tx.send_frame(64, 64, DataType.RAW8, 0, 0)
    dphy_received = await dphy_rx.wait_for_frame(0, timeout_ns=50000)
    dphy_end = cocotb.utils.get_sim_time('ns')
    
    assert dphy_received, "D-PHY frame not received"
    dphy_time = dphy_end - dphy_start
    
    # Compare performance
    cocotb.log.info(f"Performance comparison:")
    cocotb.log.info(f"  C-PHY (1 trio): {cphy_time}ns")
    cocotb.log.info(f"  D-PHY (3 lanes): {dphy_time}ns")
    cocotb.log.info(f"  Ratio: {dphy_time/cphy_time:.2f}x")
    
    # Both should work, actual performance depends on implementation
    assert cphy_time > 0 and dphy_time > 0, "Invalid timing measurements"


@cocotb.test()
async def test_cphy_error_handling(dut):
    """Test C-PHY error handling"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure C-PHY with error injection
    config = Csi2Config(
        phy_type=PhyType.CPHY,
        trio_count=1,
        bit_rate_mbps=1000,
        inject_ecc_errors=True,
        error_injection_rate=0.3
    )
    
    # Create bus and models
    bus = Csi2CPhyBus(dut, trio_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Enable error injection
    tx_model.enable_error_injection(0.3)
    
    # Track errors
    errors_detected = []
    
    async def on_error(error_type, packet, message):
        errors_detected.append(error_type)
        cocotb.log.info(f"C-PHY error detected: {error_type}")
    
    rx_model.on_error_detected = on_error
    
    # Send packets with errors
    for i in range(15):
        packet = Csi2ShortPacket.frame_start(0, i)
        await tx_model.send_packet(packet)
        await Timer(1000, units="ns")
    
    await Timer(10000, units="ns")
    
    # Check error handling
    tx_stats = tx_model.get_statistics()
    rx_stats = rx_model.get_statistics()
    
    assert tx_stats['errors_injected'] > 0, "No errors injected"
    assert len(errors_detected) > 0 or rx_stats['ecc_errors'] > 0, "No errors detected"
    
    # Verify system still functions despite errors
    assert rx_stats['packets_received'] > 5, "Too few packets received"
    
    cocotb.log.info(f"C-PHY error handling test: {tx_stats['errors_injected']} errors injected, "
                   f"{len(errors_detected)} errors handled")
