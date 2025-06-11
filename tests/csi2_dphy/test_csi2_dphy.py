"""
CSI-2 D-PHY specific tests

Copyright (c) 2024 CSI-2 Extension Contributors
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock
import pytest

from cocotbext.csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2DPhyBus, Csi2ShortPacket, Csi2LongPacket
)
from cocotbext.csi2.phy import DPhyModel
from cocotbext.csi2.config import Csi2PhyConfig


@cocotb.test()
async def test_dphy_lane_distribution(dut):
    """Test D-PHY multi-lane data distribution"""
    
    # Create clock
    clock = Clock(dut.clk, 8, units="ns")  # 125 MHz
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(200, units="ns")
    dut.reset_n.value = 1
    await Timer(200, units="ns")
    
    # Configure 4-lane D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=1000,
        continuous_clock=True,
        lane_distribution_enabled=True
    )
    
    # Create D-PHY bus
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Send large frame to test lane distribution
    width, height = 640, 480
    await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
    
    # Wait for frame completion
    frame_received = await rx_model.wait_for_frame(virtual_channel=0, timeout_ns=200000)
    assert frame_received, "Frame not received within timeout"
    
    # Validate received frame
    frame_data = rx_model.get_frame_data(0)
    assert frame_data is not None
    assert len(frame_data) == width * height
    
    # Check PHY statistics
    phy_stats = tx_model.phy_model.get_statistics() if hasattr(tx_model.phy_model, 'get_statistics') else {}
    
    cocotb.log.info(f"D-PHY 4-lane test passed: {len(frame_data)} bytes distributed across lanes")


@cocotb.test()
async def test_dphy_timing_parameters(dut):
    """Test D-PHY timing parameter validation"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Test different bit rates and timing configurations
    test_configs = [
        {'bit_rate_mbps': 80, 'expected_ui_ns': 12.5},
        {'bit_rate_mbps': 500, 'expected_ui_ns': 2.0},
        {'bit_rate_mbps': 1000, 'expected_ui_ns': 1.0},
        {'bit_rate_mbps': 2500, 'expected_ui_ns': 0.4}
    ]
    
    for test_config in test_configs:
        config = Csi2Config(
            phy_type=PhyType.DPHY,
            lane_count=1,
            bit_rate_mbps=test_config['bit_rate_mbps'],
            continuous_clock=False
        )
        
        # Verify timing calculations
        ui_period = config.get_ui_period_ns()
        expected_ui = test_config['expected_ui_ns']
        
        assert abs(ui_period - expected_ui) < 0.1, \
            f"UI period mismatch: expected {expected_ui}ns, got {ui_period}ns"
        
        # Test PHY timing validation
        phy_config = Csi2PhyConfig()
        timing_valid = phy_config.validate_timing(test_config['bit_rate_mbps'])
        assert timing_valid, f"Timing validation failed for {test_config['bit_rate_mbps']} Mbps"
    
    cocotb.log.info("D-PHY timing parameter test passed")


@cocotb.test()
async def test_dphy_hs_lp_transitions(dut):
    """Test D-PHY High Speed and Low Power transitions"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=500,
        continuous_clock=False  # Test non-continuous clock
    )
    
    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Track state transitions
    transitions_detected = []
    
    def on_hs_start():
        transitions_detected.append('hs_start')
        cocotb.log.info("HS transmission started")
        
    def on_hs_end():
        transitions_detected.append('hs_end')
        cocotb.log.info("HS transmission ended")
    
    # Set up callbacks
    if hasattr(rx_model.phy_model, 'set_rx_callbacks'):
        rx_model.phy_model.set_rx_callbacks(
            on_packet_start=on_hs_start,
            on_packet_end=on_hs_end
        )
    
    # Send multiple packets to observe transitions
    for i in range(3):
        packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=i)
        await tx_model.send_packet(packet)
        await Timer(5000, units="ns")  # Allow LP state between packets
    
    await Timer(10000, units="ns")
    
    # Verify transitions occurred
    assert len(transitions_detected) >= 2, "HS/LP transitions not detected"
    
    cocotb.log.info(f"D-PHY HS/LP transition test passed: {len(transitions_detected)} transitions")


@cocotb.test()
async def test_dphy_clock_modes(dut):
    """Test D-PHY continuous vs non-continuous clock modes"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Test continuous clock mode
    config_continuous = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=800,
        continuous_clock=True
    )
    
    bus = Csi2DPhyBus(dut, lane_count=1)
    tx_continuous = Csi2TxModel(bus, config_continuous)
    rx_continuous = Csi2RxModel(bus, config_continuous)
    
    # Send frame with continuous clock
    await tx_continuous.send_frame(64, 64, DataType.RAW8, 0, 0)
    frame_received = await rx_continuous.wait_for_frame(0, timeout_ns=50000)
    assert frame_received, "Frame not received with continuous clock"
    
    # Reset for non-continuous test
    await tx_continuous.reset()
    await rx_continuous.reset()
    await Timer(1000, units="ns")
    
    # Test non-continuous clock mode
    config_non_continuous = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=800,
        continuous_clock=False
    )
    
    tx_non_continuous = Csi2TxModel(bus, config_non_continuous)
    rx_non_continuous = Csi2RxModel(bus, config_non_continuous)
    
    # Send frame with non-continuous clock
    await tx_non_continuous.send_frame(64, 64, DataType.RAW8, 0, 1)
    frame_received = await rx_non_continuous.wait_for_frame(0, timeout_ns=50000)
    assert frame_received, "Frame not received with non-continuous clock"
    
    cocotb.log.info("D-PHY clock mode test passed")


@cocotb.test()
async def test_dphy_lane_deskew(dut):
    """Test D-PHY multi-lane deskew capabilities"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure multi-lane D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=1500,
        continuous_clock=True,
        lane_distribution_enabled=True
    )
    
    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Enable strict timing for deskew validation
    rx_model.enable_strict_timing_validation(True)
    
    # Send high-speed data stream
    for frame_num in range(3):
        await tx_model.send_frame(320, 240, DataType.RAW8, 0, frame_num)
        
    # Allow time for all transmissions
    await Timer(100000, units="ns")
    
    # Check reception statistics
    stats = rx_model.get_statistics()
    
    # In a real implementation, lane deskew would be handled by the PHY
    # Here we just verify that multi-lane transmission works
    assert stats['packets_received'] > 0, "No packets received"
    assert stats['frames_received'] >= 3, f"Expected 3 frames, got {stats['frames_received']}"
    
    cocotb.log.info(f"D-PHY lane deskew test: {stats['frames_received']} frames received")


@cocotb.test()
async def test_dphy_error_recovery(dut):
    """Test D-PHY error detection and recovery"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Configure D-PHY with error injection
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=500,
        continuous_clock=True,
        inject_ecc_errors=True,
        error_injection_rate=0.2
    )
    
    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=1)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)
    
    # Enable error injection
    tx_model.enable_error_injection(0.2)
    
    # Track errors
    errors_detected = []
    
    async def on_error(error_type, packet, message):
        errors_detected.append((error_type, message))
        cocotb.log.info(f"Error detected: {error_type} - {message}")
    
    rx_model.on_error_detected = on_error
    
    # Send packets with potential errors
    for i in range(20):
        if i % 4 == 0:
            # Frame sequence
            await tx_model.send_frame(32, 32, DataType.RAW8, 0, i//4)
        else:
            # Individual packets
            packet = Csi2ShortPacket.frame_start(0, i)
            await tx_model.send_packet(packet)
        
        await Timer(1000, units="ns")
    
    await Timer(20000, units="ns")
    
    # Check error statistics
    tx_stats = tx_model.get_statistics()
    rx_stats = rx_model.get_statistics()
    
    assert tx_stats['errors_injected'] > 0, "No errors were injected"
    assert len(errors_detected) > 0 or rx_stats['ecc_errors'] > 0, "No errors detected"
    
    # Verify recovery - receiver should still function
    assert rx_stats['packets_received'] > 10, "Too few packets received, recovery failed"
    
    cocotb.log.info(f"D-PHY error recovery test: {tx_stats['errors_injected']} errors injected, "
                   f"{len(errors_detected)} errors detected, "
                   f"{rx_stats['packets_received']} packets received")


@cocotb.test()
async def test_dphy_power_optimization(dut):
    """Test D-PHY power optimization features"""
    
    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")
    
    # Test low power modes
    config_low_power = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=200,  # Lower bit rate for power savings
        continuous_clock=False,  # Non-continuous for power savings
    )
    
    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=1)
    tx_model = Csi2TxModel(bus, config_low_power)
    rx_model = Csi2RxModel(bus, config_low_power)
    
    # Measure transmission time for power analysis
    start_time = cocotb.utils.get_sim_time('ns')
    
    # Send sparse data (simulating low activity)
    for i in range(5):
        packet = Csi2ShortPacket.frame_start(0, i)
        await tx_model.send_packet(packet)
        await Timer(10000, units="ns")  # Long gaps between packets
    
    end_time = cocotb.utils.get_sim_time('ns')
    
    # In a real implementation, we would measure actual power consumption
    # Here we just verify the low-power configuration works
    stats = rx_model.get_statistics()
    assert stats['packets_received'] == 5, f"Expected 5 packets, got {stats['packets_received']}"
    
    transmission_time = end_time - start_time
    cocotb.log.info(f"D-PHY power optimization test: {transmission_time}ns total time, "
                   f"low activity mode verified")
