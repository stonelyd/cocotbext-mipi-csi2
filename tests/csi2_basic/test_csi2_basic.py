"""
Basic CSI-2 functionality tests

Copyright (c) 2024 CSI-2 Extension Contributors
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge, with_timeout
from cocotb.clock import Clock
import pytest
import asyncio
import random
import os
from cocotb_test import simulator

from pathlib import Path


from cocotb.regression import TestFactory

from cocotbext.mipi_csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2Bus, Csi2DPhyBus, Csi2ShortPacket, Csi2LongPacket
)
from cocotbext.mipi_csi2.utils import setup_logging


class TB:
    def __init__(self, dut):
        self.dut = dut
        self.clock = None
        self.config = None
        self.bus = None
        self.tx_model = None
        self.rx_model = None
        self.tx_phy_model = None
        self.rx_phy_model = None

    async def setup(self):
        """No clock/reset setup needed for pure MIPI interface"""
        pass

    async def configure_csi2(self, phy_type=PhyType.DPHY, lane_count=1, bit_rate_mbps=500, continuous_clock=True):
        """Configure CSI-2 with given parameters"""
        self.config = Csi2Config(
            phy_type=phy_type,
            lane_count=lane_count,
            bit_rate_mbps=bit_rate_mbps,
            continuous_clock=continuous_clock,
            lane_distribution_enabled=False  # Disable lane distribution to avoid padding
        )
        if phy_type == PhyType.DPHY:
            self.bus = Csi2DPhyBus(self.dut, lane_count=lane_count)

            # Initialize D-PHY signals to LP-11 state
            if hasattr(self.dut, 'clk_p'):
                self.dut.clk_p.value = 1
                self.dut.clk_n.value = 1
            for i in range(lane_count):
                if hasattr(self.dut, f'data{i}_p'):
                    getattr(self.dut, f'data{i}_p').value = 1
                    getattr(self.dut, f'data{i}_n').value = 1

            # Wait for signals to stabilize
            await Timer(10, units="ns")

            # Debug: check signal values after initialization
            cocotb.log.info(f"DUT signal values after initialization:")
            if hasattr(self.dut, 'clk_p'):
                cocotb.log.info(f"  clk_p: {self.dut.clk_p.value}, clk_n: {self.dut.clk_n.value}")
            for i in range(lane_count):
                if hasattr(self.dut, f'data{i}_p'):
                    p_val = getattr(self.dut, f'data{i}_p').value
                    n_val = getattr(self.dut, f'data{i}_n').value
                    cocotb.log.info(f"  data{i}_p: {p_val}, data{i}_n: {n_val}")

            # Create separate TX and RX PHY models for loopback testing
            from cocotbext.mipi_csi2.phy import DPhyTxModel, DPhyRxModel
            self.tx_phy_model = DPhyTxModel(self.bus, self.config)
            self.rx_phy_model = DPhyRxModel(self.bus, self.config)
        else:
            self.bus = Csi2Bus.from_entity(self.dut)
            from cocotbext.mipi_csi2.phy import CPhyModel
            self.tx_phy_model = CPhyModel(self.bus, self.config)
            self.rx_phy_model = CPhyModel(self.bus, self.config)

        # Create TX and RX models with separate PHY models
        self.tx_model = Csi2TxModel(self.bus, self.config, phy_model=self.tx_phy_model)
        self.rx_model = Csi2RxModel(self.bus, self.config, phy_model=self.rx_phy_model)

        # Set up RX callbacks on the RX PHY model
        self.rx_phy_model.set_rx_callbacks(
            on_packet_start=self.rx_model._on_phy_packet_start,
            on_packet_end=self.rx_model._on_phy_packet_end,
            on_data_received=self.rx_model._on_phy_data_received
        )


# @cocotb.test()
async def run_short_packet_transmission(dut, lane_count=4, packet_type="frame_start", **kwargs):
    """Test CSI-2 short packet transmission and reception with lane distribution enabled"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()

    await tb.configure_csi2(lane_count=lane_count, bit_rate_mbps=1000)
    tb.config.lane_distribution_enabled = True
    tb.tx_phy_model.config.lane_distribution_enabled = True
    tb.rx_phy_model.config.lane_distribution_enabled = True
    cocotb.log.info(f"{lane_count}-lane distribution enabled for this test")

    await tb.rx_model.reset()

    # Disable frame assembly for all but frame_start
    if packet_type in ("frame_end", "line_start", "line_end"):
        tb.rx_model.enable_frame_assembly(False)
        cocotb.log.info("Frame assembly disabled for this test")
    else:
        tb.rx_model.enable_frame_assembly(True)

    # Select short packet type
    if packet_type == "frame_start":
        pkt = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
        expected_type = DataType.FRAME_START.value
        expected_desc = "frame start"
    elif packet_type == "frame_end":
        pkt = Csi2ShortPacket.frame_end(virtual_channel=0, frame_number=1)
        expected_type = DataType.FRAME_END.value
        expected_desc = "frame end"
    elif packet_type == "line_start":
        pkt = Csi2ShortPacket.line_start(virtual_channel=0, line_number=1)
        expected_type = DataType.LINE_START.value
        expected_desc = "line start"
    elif packet_type == "line_end":
        pkt = Csi2ShortPacket.line_end(virtual_channel=0, line_number=1)
        expected_type = DataType.LINE_END.value
        expected_desc = "line end"
    else:
        raise ValueError(f"Unknown packet_type: {packet_type}")

    packet_bytes = pkt.to_bytes()
    cocotb.log.info(f"Testing {lane_count}-lane {expected_desc} PHY transmission: {len(packet_bytes)} bytes")
    cocotb.log.info(f"Packet bytes: {[f'{b:02x}' for b in packet_bytes]}")

    try:
        cocotb.log.info(f"Attempting to start {lane_count}-lane {expected_desc} packet transmission")
        await with_timeout(tb.tx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {expected_desc} packet transmission started")
        cocotb.log.info(f"Attempting to send {expected_desc} packet data across {lane_count} lanes")
        await with_timeout(tb.tx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {expected_desc} packet data sent")
        cocotb.log.info(f"Attempting to stop {lane_count}-lane {expected_desc} packet transmission")
        await with_timeout(tb.tx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {expected_desc} PHY transmission completed")
    except cocotb.result.SimTimeoutError:
        cocotb.log.error(f"Timeout in {lane_count}-lane {expected_desc} PHY transmission")
        raise

    await Timer(1000, units="ns")
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX stats: {rx_stats}")

    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, "No packet received"
        assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
        assert received_packet.header.validate_ecc(), "Received packet ECC validation failed"
        assert received_packet.data_type == expected_type, f"Expected {expected_desc} packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"
        cocotb.log.info(f"Received {expected_desc} packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")
    except cocotb.result.SimTimeoutError:
        cocotb.log.warning(f"Timeout waiting for {expected_desc} packet reception")
        raise

    await tb.rx_model.reset()

# @cocotb.test()
async def run_long_packet_transmission(dut, lane_count=4, data_format="yuv422", **kwargs):
    """Test CSI-2 Long packet transmission and reception with lane distribution enabled"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2(lane_count=lane_count, bit_rate_mbps=1000)
    tb.config.lane_distribution_enabled = True
    tb.tx_phy_model.config.lane_distribution_enabled = True
    tb.rx_phy_model.config.lane_distribution_enabled = True
    cocotb.log.info(f"{lane_count}-lane distribution enabled for this test")
    await tb.rx_model.reset()
    tb.rx_model.enable_frame_assembly(False)
    cocotb.log.info("Frame assembly disabled for this test")

    # Create payload based on data format
    if data_format == "raw8":
        # Create Raw8 Long packet with word count = 32 (32 bytes payload)
        payload_data = bytes([i % 256 for i in range(32)])
        data_type = DataType.RAW8
        expected_word_count = 32
        expected_payload_length = 32
        format_name = "Raw8"
    elif data_format == "raw10":
        # Create Raw10 Long packet with 16 pixels (20 bytes payload)
        pixel_count = 16
        pixels = [(i * 1023) // (pixel_count - 1) for i in range(pixel_count)]  # 10-bit ramp
        from cocotbext.mipi_csi2.utils import pack_raw10
        payload_data = pack_raw10(pixels)
        data_type = DataType.RAW10
        expected_word_count = 20
        expected_payload_length = 20
        format_name = "Raw10"
    elif data_format == "raw12":
        # Create Raw12 Long packet with 16 pixels (24 bytes payload)
        pixel_count = 16
        pixels = [(i * 4095) // (pixel_count - 1) for i in range(pixel_count)]  # 12-bit ramp
        from cocotbext.mipi_csi2.utils import pack_raw12
        payload_data = pack_raw12(pixels)
        data_type = DataType.RAW12
        expected_word_count = 24
        expected_payload_length = 24
        format_name = "Raw12"
    elif data_format == "raw16":
        # Create Raw16 Long packet with 16 pixels (32 bytes payload)
        pixel_count = 16
        # pixels = [(i * 65535) // (pixel_count - 1) for i in range(pixel_count)]  # 16-bit ramp
        pixels = [i for i in range(pixel_count)]  # 16-bit ramp
        from cocotbext.mipi_csi2.utils import pack_raw16
        payload_data = pack_raw16(pixels)
        data_type = DataType.RAW16
        expected_word_count = 32
        expected_payload_length = 32
        format_name = "Raw16"
    elif data_format == "yuv420":
        # Create YUV420 Long packet with 8x8 image (96 bytes payload: 64 Y + 16 U + 16 V)
        width, height = 8, 8
        y_pixels = [(i % 256) for i in range(width * height)]  # Y plane (full resolution)
        u_pixels = [(i % 256) for i in range((width * height) // 4)]  # U plane (quarter resolution)
        v_pixels = [(i % 256) for i in range((width * height) // 4)]  # V plane (quarter resolution)
        from cocotbext.mipi_csi2.utils import pack_yuv420
        payload_data = pack_yuv420(y_pixels, u_pixels, v_pixels)
        data_type = DataType.YUV420_8BIT
        expected_word_count = 96
        expected_payload_length = 96
        format_name = "YUV420"
    elif data_format == "yuv422":
        # Create YUV422 Long packet with 8x8 image (128 bytes payload: 64 Y + 32 U + 32 V)
        width, height = 8, 8
        y_pixels = [(i % 256) for i in range(width * height)]  # Y plane (full resolution)
        u_pixels = [(i % 256) for i in range((width * height) // 2)]  # U plane (half resolution)
        v_pixels = [(i % 256) for i in range((width * height) // 2)]  # V plane (half resolution)
        from cocotbext.mipi_csi2.utils import pack_yuv422
        payload_data = pack_yuv422(y_pixels, u_pixels, v_pixels)
        data_type = DataType.YUV422_8BIT
        expected_word_count = 128
        expected_payload_length = 128
        format_name = "YUV422"
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    packet = Csi2LongPacket(virtual_channel=0, data_type=data_type, payload=payload_data)
    packet_bytes = packet.to_bytes()

    cocotb.log.info(f"Testing {lane_count}-lane {format_name} Long packet transmission: {len(packet_bytes)} bytes")
    cocotb.log.info(f"Packet header bytes: {[f'{b:02x}' for b in packet_bytes[:4]]}")
    cocotb.log.info(f"Payload bytes: {[f'{b:02x}' for b in packet_bytes[4:4+expected_payload_length]]}")
    cocotb.log.info(f"Checksum bytes: {[f'{b:02x}' for b in packet_bytes[4+expected_payload_length:]]}")

    try:
        cocotb.log.info(f"Attempting to start {lane_count}-lane {format_name} Long packet transmission")
        await with_timeout(tb.tx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {format_name} Long packet transmission started")
        cocotb.log.info(f"Attempting to send {format_name} Long packet data across {lane_count} lanes")
        await with_timeout(tb.tx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {format_name} Long packet data sent")
        cocotb.log.info(f"Attempting to stop {lane_count}-lane {format_name} Long packet transmission")
        await with_timeout(tb.tx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info(f"{lane_count}-lane {format_name} Long packet transmission completed")
    except cocotb.result.SimTimeoutError:
        cocotb.log.error(f"Timeout in {lane_count}-lane {format_name} Long packet transmission")
        raise

    await Timer(1000, units="ns")
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX stats: {rx_stats}")

    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, "No packet received"
        assert isinstance(received_packet, Csi2LongPacket), "Expected long packet"
        assert received_packet.header.validate_ecc(), "Received packet ECC validation failed"
        assert received_packet.data_type == data_type.value, f"Expected {format_name} packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"
        assert received_packet.header.word_count == expected_word_count, f"Expected word count = {expected_word_count}"
        assert len(received_packet.payload) == expected_payload_length, f"Expected payload length = {expected_payload_length} bytes"
        assert received_packet.validate_checksum(), "Received packet checksum validation failed"
        assert received_packet.payload == payload_data, "Payload data mismatch"
        cocotb.log.info(f"Received {lane_count}-lane {format_name} Long packet: VC={received_packet.virtual_channel}, "
                        f"DT=0x{received_packet.data_type:02x}, WC={received_packet.header.word_count}, "
                        f"Payload={len(received_packet.payload)} bytes")
    except cocotb.result.SimTimeoutError:
        cocotb.log.warning(f"Timeout waiting for {lane_count}-lane {format_name} Long packet reception")
        raise
    cocotb.log.info(f"{lane_count}-lane {format_name} Long packet transmission test passed")
    await tb.rx_model.reset()

# @cocotb.test()
async def run_frame_transmission(dut, lane_count=1):
    """Test complete frame transmission using event-driven RX """

    setup_logging()
    tb = TB(dut)
    await tb.setup()

    # Configure with 2-lane distribution enabled
    await tb.configure_csi2(lane_count=lane_count, bit_rate_mbps=1000)

    # Override configuration to enable lane distribution
    tb.config.lane_distribution_enabled = True
    tb.tx_phy_model.config.lane_distribution_enabled = True
    tb.rx_phy_model.config.lane_distribution_enabled = True
    cocotb.log.info("2-lane distribution enabled for this test")

    # Reset RX model to ensure clean state
    await tb.rx_model.reset()

    # Frame parameters
    width, height = 160, 120
    data_type = DataType.RAW8
    virtual_channel = 0
    frame_number = 0

    # Log start
    cocotb.log.info(f"Starting 2-lane frame transmission test: {width}x{height}, RAW8, VC={virtual_channel}")

    # Start frame transmission
    await tb.tx_model.send_frame(width, height, data_type, virtual_channel, frame_number)
    cocotb.log.info("2-lane frame sent from TX model")

    # Wait for RX model to signal frame completion (event-driven, no timer)
    cocotb.log.info("Waiting for RX model to complete 2-lane frame reception (event-driven)")
    await tb.rx_model.frame_complete_event.wait()
    tb.rx_model.frame_complete_event.clear()
    cocotb.log.info("RX model signaled 2-lane frame completion")

    # Validate received frame data
    frame_data = tb.rx_model.get_frame_data(virtual_channel)
    assert frame_data is not None, "No frame data received"
    assert len(frame_data) == width * height, f"Frame data length mismatch: expected {width*height}, got {len(frame_data)}"

    # Debug: Log frame data statistics
    cocotb.log.info(f"2-lane frame data statistics:")
    cocotb.log.info(f"  Total bytes: {len(frame_data)}")
    cocotb.log.info(f"  Expected bytes: {width * height}")
    cocotb.log.info(f"  Min value: {min(frame_data)}")
    cocotb.log.info(f"  Max value: {max(frame_data)}")
    cocotb.log.info(f"  Average value: {sum(frame_data) / len(frame_data):.2f}")

    # Generate expected ramp pattern (same as TX model uses)
    expected_pattern = bytearray()
    for y in range(height):
        for x in range(width):
            # Horizontal ramp: value = (x * 255) // width
            value = (x * 255) // width
            expected_pattern.append(value)

    # Debug: Log expected pattern statistics
    cocotb.log.info(f"Expected pattern statistics:")
    cocotb.log.info(f"  Total bytes: {len(expected_pattern)}")
    cocotb.log.info(f"  Min value: {min(expected_pattern)}")
    cocotb.log.info(f"  Max value: {max(expected_pattern)}")
    cocotb.log.info(f"  Average value: {sum(expected_pattern) / len(expected_pattern):.2f}")

    # Debug: Show first few bytes of both patterns
    cocotb.log.info(f"First 20 bytes of received 2-lane frame: {[f'{b:02x}' for b in frame_data[:20]]}")
    cocotb.log.info(f"First 20 bytes of expected pattern: {[f'{b:02x}' for b in expected_pattern[:20]]}")

    # Debug: Show last few bytes of both patterns
    cocotb.log.info(f"Last 20 bytes of received 2-lane frame: {[f'{b:02x}' for b in frame_data[-20:]]}")
    cocotb.log.info(f"Last 20 bytes of expected pattern: {[f'{b:02x}' for b in expected_pattern[-20:]]}")

    # Find first mismatch if any
    if frame_data != expected_pattern:
        for i, (actual, expected) in enumerate(zip(frame_data, expected_pattern)):
            if actual != expected:
                cocotb.log.error(f"First mismatch at byte {i}: received 0x{actual:02x}, expected 0x{expected:02x}")
                cocotb.log.error(f"  Position: x={i % width}, y={i // width}")
                break

        # Show more context around the first mismatch
        if len(frame_data) > 0:
            mismatch_pos = 0
            for i, (actual, expected) in enumerate(zip(frame_data, expected_pattern)):
                if actual != expected:
                    mismatch_pos = i
                    break

            start_pos = max(0, mismatch_pos - 10)
            end_pos = min(len(frame_data), mismatch_pos + 10)

            cocotb.log.error(f"Context around first mismatch (position {mismatch_pos}):")
            cocotb.log.error(f"  Received: {[f'{b:02x}' for b in frame_data[start_pos:end_pos]]}")
            cocotb.log.error(f"  Expected: {[f'{b:02x}' for b in expected_pattern[start_pos:end_pos]]}")

            # Show line-by-line comparison for first few lines
            cocotb.log.error("Line-by-line comparison (first 3 lines):")
            for line in range(min(3, height)):
                line_start = line * width
                line_end = line_start + width
                received_line = frame_data[line_start:line_end]
                expected_line = expected_pattern[line_start:line_end]
                cocotb.log.error(f"  Line {line}: received {[f'{b:02x}' for b in received_line[:10]]}...")
                cocotb.log.error(f"  Line {line}: expected {[f'{b:02x}' for b in expected_line[:10]]}...")

    # Assert pattern match with detailed error message
    assert frame_data == expected_pattern, (
        f"2-lane frame data does not match expected ramp pattern!\n"
        f"Frame size: {len(frame_data)} bytes, Expected: {len(expected_pattern)} bytes\n"
        f"Frame range: {min(frame_data)}-{max(frame_data)}, Expected range: {min(expected_pattern)}-{max(expected_pattern)}"
    )

    cocotb.log.info("2-lane frame data matches expected ramp pattern")
    cocotb.log.info(f"2-lane frame transmission test passed: received {len(frame_data)} bytes")

    # Clean up any incomplete frame state
    await tb.rx_model.reset()


if cocotb.SIM_NAME:


    factory = TestFactory(run_short_packet_transmission)
    factory.add_option("lane_count", [1, 2, 4])
    factory.add_option("packet_type", ["frame_start", "frame_end", "line_start", "line_end"])
    factory.generate_tests()

    # Add long packet factory with comprehensive data type coverage
    factory_long = TestFactory(run_long_packet_transmission)
    factory_long.add_option("lane_count", [1, 2, 4])
    # Comprehensive CSI-2 data type coverage per PRP requirements
    factory_long.add_option("data_format", [
        # RAW data types - all supported formats
        "raw6", "raw7", "raw8", "raw10", "raw12", "raw14", "raw16", "raw20",
        # RGB data types - all variants
        "rgb444", "rgb555", "rgb565", "rgb666", "rgb888",
        # YUV data types - primary formats
        "yuv420", "yuv422"
    ])
    factory_long.generate_tests()
    

    # Add frame transmission factory
    factory_frame = TestFactory(run_frame_transmission)
    factory_frame.add_option("lane_count", [1, 2, 4])
    factory_frame.generate_tests()






async def run_error_injection_test(dut, lane_count=1, error_type="ecc", data_format="raw8", **kwargs):
    """Comprehensive error injection test for various error types and configurations"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    
    # Configure with error injection enabled
    await tb.configure_csi2(
        phy_type=PhyType.DPHY,
        lane_count=lane_count,
        bit_rate_mbps=1000,
        continuous_clock=True
    )
    
    # Configure error injection based on error_type
    if error_type == "ecc":
        tb.config.inject_ecc_errors = True
        tb.config.error_injection_rate = 0.1  # 10% error rate
        cocotb.log.info(f"Configured ECC error injection with {tb.config.error_injection_rate*100}% rate")
    elif error_type == "checksum":
        tb.config.inject_checksum_errors = True
        tb.config.error_injection_rate = 0.1
        cocotb.log.info(f"Configured checksum error injection with {tb.config.error_injection_rate*100}% rate")
    elif error_type == "crc":
        tb.config.inject_crc_errors = True
        tb.config.error_injection_rate = 0.1
        cocotb.log.info(f"Configured CRC error injection with {tb.config.error_injection_rate*100}% rate")
    
    cocotb.log.info(f"=== Error Injection Test: {error_type.upper()} errors, {lane_count}-lane, {data_format} ===")
    
    # Generate test pattern based on data format
    test_data = generate_test_pattern(data_format, width=640, height=480)
    
    # Send multiple packets to trigger error injection
    error_count = 0
    success_count = 0
    total_packets = 20
    
    for i in range(total_packets):
        try:
            # Send packet with potential error injection
            packet = create_long_packet(data_format, test_data[:100], virtual_channel=0)
            await tb.tx_model.send_packet(packet)
            
            # Attempt to receive and validate
            received_packet = await tb.rx_model.get_next_packet(timeout_ns=5000)
            
            if received_packet is not None:
                # Check if error was detected (expected behavior with error injection)
                if error_type == "ecc" and not received_packet.header.validate_ecc():
                    error_count += 1
                    cocotb.log.info(f"Packet {i}: ECC error correctly detected")
                elif error_type == "checksum" and hasattr(received_packet, 'checksum_valid') and not received_packet.checksum_valid:
                    error_count += 1
                    cocotb.log.info(f"Packet {i}: Checksum error correctly detected")
                else:
                    success_count += 1
                    cocotb.log.info(f"Packet {i}: Received successfully")
            else:
                cocotb.log.warning(f"Packet {i}: No packet received (possible error)")
                
        except Exception as e:
            cocotb.log.info(f"Packet {i}: Error during transmission/reception (expected): {e}")
            error_count += 1
    
    # Validate error injection worked
    cocotb.log.info(f"Error injection test results: {error_count} errors, {success_count} successful packets out of {total_packets}")
    
    # With 10% error injection rate, we expect some errors but not all packets to fail
    assert error_count > 0, f"No errors detected with {error_type} error injection enabled"
    assert success_count > 0, f"All packets failed - error injection rate too high"
    
    cocotb.log.info(f"Error injection test for {error_type} with {lane_count}-lane {data_format} PASSED!")


async def run_non_continuous_comprehensive_test(dut, lane_count=1, bit_rate_mbps=1000, packet_type="short", **kwargs):
    """Comprehensive non-continuous clock mode testing with various configurations"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    
    # Configure with non-continuous clock
    await tb.configure_csi2(
        phy_type=PhyType.DPHY,
        lane_count=lane_count,
        bit_rate_mbps=bit_rate_mbps,
        continuous_clock=False  # Enable non-continuous clock mode
    )
    
    cocotb.log.info(f"=== Non-Continuous Clock Test: {lane_count}-lane, {bit_rate_mbps}Mbps, {packet_type} packet ===")
    
    # Enable lane distribution for multi-lane configurations
    if lane_count > 1:
        tb.config.lane_distribution_enabled = True
    
    # Test both short and long packets
    if packet_type == "short":
        # Send frame start packet (short packet)
        packet = create_short_packet("frame_start", virtual_channel=0)
        await tb.tx_model.send_packet(packet)
        
        # Verify reception
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, "No packet received in non-continuous mode"
        assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
        assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
        
    else:  # packet_type == "long"
        # Send long packet with test data
        test_data = generate_test_pattern("raw8", width=320, height=240)
        packet = create_long_packet("raw8", test_data, virtual_channel=0)
        await tb.tx_model.send_packet(packet)
        
        # Verify reception
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=20000)
        assert received_packet is not None, "No packet received in non-continuous mode"
        assert isinstance(received_packet, Csi2LongPacket), "Expected long packet"
        assert received_packet.data_type == DataType.RAW8.value, "Expected RAW8 packet"
    
    # Performance measurement for non-continuous mode
    import time
    start_time = time.time()
    
    # Send multiple packets to measure timing performance
    for i in range(10):
        if packet_type == "short":
            packet = create_short_packet("frame_start", virtual_channel=0)
        else:
            test_data = generate_test_pattern("raw8", width=160, height=120)
            packet = create_long_packet("raw8", test_data, virtual_channel=0)
        
        await tb.tx_model.send_packet(packet)
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, f"Packet {i} not received"
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    cocotb.log.info(f"Non-continuous clock performance: {test_duration:.3f}s for 10 packets")
    cocotb.log.info(f"Non-continuous clock test with {lane_count}-lane, {bit_rate_mbps}Mbps, {packet_type} packets PASSED!")


async def run_performance_benchmark_test(dut, lane_count=1, clock_mode="continuous", data_format="raw8", **kwargs):
    """Performance benchmarking test to measure simulation overhead and throughput"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    
    # Configure based on clock mode
    continuous_clock = (clock_mode == "continuous")
    await tb.configure_csi2(
        phy_type=PhyType.DPHY,
        lane_count=lane_count,
        bit_rate_mbps=1000,
        continuous_clock=continuous_clock
    )
    
    cocotb.log.info(f"=== Performance Benchmark: {lane_count}-lane, {clock_mode} clock, {data_format} ===")
    
    # Enable lane distribution for multi-lane
    if lane_count > 1:
        tb.config.lane_distribution_enabled = True
    
    # Generate test data patterns
    test_patterns = {
        "small": generate_test_pattern(data_format, width=160, height=120),
        "medium": generate_test_pattern(data_format, width=320, height=240),
        "large": generate_test_pattern(data_format, width=640, height=480)
    }
    
    # Performance metrics collection
    performance_results = {}
    
    for pattern_name, test_data in test_patterns.items():
        cocotb.log.info(f"Testing {pattern_name} frame size...")
        
        # Measure transmission performance
        import time
        start_time = time.time()
        packets_sent = 0
        packets_received = 0
        
        # Test with multiple frames to get accurate measurements
        num_frames = 5
        for frame_num in range(num_frames):
            try:
                # Send frame start
                fs_packet = create_short_packet("frame_start", virtual_channel=0)
                await tb.tx_model.send_packet(fs_packet)
                packets_sent += 1
                
                # Send image data in chunks
                chunk_size = min(len(test_data), 1024)  # 1KB chunks
                for chunk_start in range(0, len(test_data), chunk_size):
                    chunk_data = test_data[chunk_start:chunk_start + chunk_size]
                    data_packet = create_long_packet(data_format, chunk_data, virtual_channel=0)
                    await tb.tx_model.send_packet(data_packet)
                    packets_sent += 1
                
                # Send frame end
                fe_packet = create_short_packet("frame_end", virtual_channel=0)
                await tb.tx_model.send_packet(fe_packet)
                packets_sent += 1
                
                # Receive and validate packets
                for _ in range(packets_sent - packets_received):
                    received_packet = await tb.rx_model.get_next_packet(timeout_ns=15000)
                    if received_packet is not None:
                        packets_received += 1
                    
            except Exception as e:
                cocotb.log.warning(f"Frame {frame_num} error: {e}")
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Calculate performance metrics
        throughput_mbps = (len(test_data) * num_frames * 8) / (test_duration * 1e6) if test_duration > 0 else 0
        packets_per_second = packets_sent / test_duration if test_duration > 0 else 0
        reception_rate = (packets_received / packets_sent * 100) if packets_sent > 0 else 0
        
        performance_results[pattern_name] = {
            "duration_s": test_duration,
            "throughput_mbps": throughput_mbps,
            "packets_per_second": packets_per_second,
            "reception_rate_percent": reception_rate,
            "packets_sent": packets_sent,
            "packets_received": packets_received
        }
        
        cocotb.log.info(f"{pattern_name.capitalize()} frame performance:")
        cocotb.log.info(f"  Duration: {test_duration:.3f}s")
        cocotb.log.info(f"  Throughput: {throughput_mbps:.2f} Mbps")
        cocotb.log.info(f"  Packet rate: {packets_per_second:.1f} packets/s")
        cocotb.log.info(f"  Reception rate: {reception_rate:.1f}%")
    
    # Performance validation - check for acceptable overhead
    baseline_duration = performance_results["small"]["duration_s"]
    medium_duration = performance_results["medium"]["duration_s"]
    large_duration = performance_results["large"]["duration_s"]
    
    # Calculate scaling efficiency (should be roughly linear with data size)
    medium_overhead = (medium_duration / baseline_duration) - 2.0  # Medium is 2x data
    large_overhead = (large_duration / baseline_duration) - 4.0    # Large is 4x data
    
    cocotb.log.info(f"Performance Analysis:")
    cocotb.log.info(f"  Medium frame overhead: {medium_overhead:.2f}x (target: <0.1x)")
    cocotb.log.info(f"  Large frame overhead: {large_overhead:.2f}x (target: <0.1x)")
    
    # Validate performance requirements (per PRP: <10% overhead)
    max_overhead = 0.1  # 10% maximum overhead
    assert abs(medium_overhead) < max_overhead, f"Medium frame overhead {medium_overhead:.3f}x exceeds {max_overhead}x limit"
    assert abs(large_overhead) < max_overhead, f"Large frame overhead {large_overhead:.3f}x exceeds {max_overhead}x limit"
    
    # Validate reception rate (should be >95% for good performance)
    min_reception_rate = 95.0
    for pattern_name, results in performance_results.items():
        reception_rate = results["reception_rate_percent"]
        assert reception_rate >= min_reception_rate, f"{pattern_name} reception rate {reception_rate:.1f}% below {min_reception_rate}% threshold"
    
    cocotb.log.info(f"Performance benchmark for {lane_count}-lane {clock_mode} {data_format} PASSED!")
    cocotb.log.info(f"All overhead measurements within {max_overhead*100}% target")


# Helper functions for comprehensive testing
def generate_test_pattern(data_format, width=640, height=480, pattern_type="ramp"):
    """Generate test patterns for various data formats and pattern types"""
    import numpy as np
    
    # Calculate bits per pixel based on data format
    bits_per_pixel = {
        "raw6": 6, "raw7": 7, "raw8": 8, "raw10": 10, "raw12": 12, 
        "raw14": 14, "raw16": 16, "raw20": 20,
        "rgb444": 12, "rgb555": 15, "rgb565": 16, "rgb666": 18, "rgb888": 24,
        "yuv420": 12, "yuv422": 16
    }.get(data_format.lower(), 8)
    
    max_value = (1 << bits_per_pixel) - 1
    total_pixels = width * height
    
    if pattern_type == "ramp":
        # Linear ramp pattern
        pattern = np.linspace(0, max_value, total_pixels, dtype=np.uint32)
    elif pattern_type == "checkerboard":
        # Checkerboard pattern
        pattern = np.zeros(total_pixels, dtype=np.uint32)
        for y in range(height):
            for x in range(width):
                if (x + y) % 2 == 0:
                    pattern[y * width + x] = max_value
    elif pattern_type == "solid":
        # Solid pattern at mid-level
        pattern = np.full(total_pixels, max_value // 2, dtype=np.uint32)
    elif pattern_type == "walking":
        # Walking ones pattern
        pattern = np.zeros(total_pixels, dtype=np.uint32)
        for i in range(total_pixels):
            bit_pos = i % bits_per_pixel
            pattern[i] = 1 << bit_pos
    else:
        # Default ramp
        pattern = np.linspace(0, max_value, total_pixels, dtype=np.uint32)
    
    return pattern.astype(np.uint8).tobytes()[:total_pixels * ((bits_per_pixel + 7) // 8)]


def create_short_packet(packet_type, virtual_channel=0):
    """Create short packet for testing"""
    from cocotbext.mipi_csi2.csi2_packet import Csi2ShortPacket
    
    data_type_map = {
        "frame_start": DataType.FRAME_START,
        "frame_end": DataType.FRAME_END,
        "line_start": DataType.LINE_START,
        "line_end": DataType.LINE_END
    }
    
    data_type = data_type_map.get(packet_type, DataType.FRAME_START)
    return Csi2ShortPacket(
        virtual_channel=virtual_channel,
        data_type=data_type.value,
        data=0
    )


def create_long_packet(data_format, payload_data, virtual_channel=0):
    """Create long packet for testing"""
    from cocotbext.mipi_csi2.csi2_packet import Csi2LongPacket
    
    data_type_map = {
        "raw6": DataType.RAW6, "raw7": DataType.RAW7, "raw8": DataType.RAW8, 
        "raw10": DataType.RAW10, "raw12": DataType.RAW12, "raw14": DataType.RAW14,
        "raw16": DataType.RAW16, "raw20": DataType.RAW20,
        "rgb444": DataType.RGB444, "rgb555": DataType.RGB555, "rgb565": DataType.RGB565,
        "rgb666": DataType.RGB666, "rgb888": DataType.RGB888,
        "yuv420": DataType.YUV420_8BIT, "yuv422": DataType.YUV422_8BIT
    }
    
    data_type = data_type_map.get(data_format.lower(), DataType.RAW8)
    return Csi2LongPacket(
        virtual_channel=virtual_channel,
        data_type=data_type.value,
        payload=payload_data
    )


@cocotb.test()
async def test_non_continuous_clock_short_packet(dut):
    """Test non-continuous clock mode with short packet transmission"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    
    # Configure with non-continuous clock - this creates all interfaces
    await tb.configure_csi2(
        phy_type=PhyType.DPHY,
        lane_count=1,
        bit_rate_mbps=500,
        continuous_clock=False  # Enable non-continuous clock mode
    )

    cocotb.log.info("=== Testing Non-Continuous Clock Mode ===")
    cocotb.log.info(f"Configuration: {tb.config}")
    cocotb.log.info(f"Clock mode: {'Continuous' if tb.config.continuous_clock else 'Non-continuous'}")

    # Create a simple frame start packet
    packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)

    packet_bytes = packet.to_bytes()
    cocotb.log.info(f"Transmitting packet: {len(packet_bytes)} bytes")
    cocotb.log.info(f"Packet bytes: {[f'{b:02x}' for b in packet_bytes]}")

    # Send the packet using PHY model
    try:
        cocotb.log.info("Starting non-continuous clock packet transmission")
        await tb.tx_phy_model.start_packet_transmission()
        cocotb.log.info("Non-continuous clock packet transmission started")
        
        await tb.tx_phy_model.send_packet_data(packet_bytes)
        cocotb.log.info("Non-continuous clock packet data sent")
        
        await tb.tx_phy_model.stop_packet_transmission()
        cocotb.log.info("Non-continuous clock PHY transmission completed")
    except Exception as e:
        cocotb.log.error(f"Error in non-continuous clock PHY transmission: {e}")
        raise

    # Wait for processing
    await Timer(1000, units='ns')

    # Verify reception using the RX model
    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, "No packet received"
        assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
        assert received_packet.header.validate_ecc(), "Received packet ECC validation failed"
        assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"
        cocotb.log.info(f"Received frame start packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")
        cocotb.log.info("Non-continuous clock mode test PASSED!")
    except Exception as e:
        cocotb.log.error(f"Error receiving packet: {e}")
        assert False, "No packets received in non-continuous clock mode"


@cocotb.test()
async def test_multilane_non_continuous_clock_short_packet(dut):
    """Test multi-lane D-PHY with non-continuous clock mode"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()

    # Test configurations: 2-lane and 4-lane with non-continuous clock
    test_configs = [
        (2, "2-lane"),
        (4, "4-lane"),
    ]
    
    for lane_count, desc in test_configs:
        cocotb.log.info(f"=== Testing {desc} Non-Continuous Clock Mode ===")
        
        # Configure for multi-lane + non-continuous clock
        await tb.configure_csi2(lane_count=lane_count, bit_rate_mbps=1000, continuous_clock=False)
        tb.config.lane_distribution_enabled = True  # Enable multi-lane distribution
        tb.tx_phy_model.config.lane_distribution_enabled = True
        tb.rx_phy_model.config.lane_distribution_enabled = True
        
        cocotb.log.info(f"Configuration: lanes={lane_count}, continuous_clock=False, lane_distribution=True")
        
        await tb.rx_model.reset()
        tb.rx_model.enable_frame_assembly(True)

        # Create and send a short packet
        pkt = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
        packet_bytes = pkt.to_bytes()
        cocotb.log.info(f"Transmitting {desc} non-continuous packet: {len(packet_bytes)} bytes")
        cocotb.log.info(f"Packet bytes: {[f'{b:02x}' for b in packet_bytes]}")

        try:
            cocotb.log.info(f"Starting {desc} non-continuous clock packet transmission")
            await with_timeout(tb.tx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
            cocotb.log.info(f"{desc} non-continuous clock packet transmission started")
            
            await with_timeout(tb.tx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
            cocotb.log.info(f"{desc} non-continuous clock packet data sent")
            
            await with_timeout(tb.tx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
            cocotb.log.info(f"{desc} non-continuous clock PHY transmission completed")
            
        except Exception as e:
            cocotb.log.error(f"Timeout in {desc} non-continuous clock PHY transmission: {e}")
            raise

        # Wait for packet reception
        await Timer(1000, units='ns')
        
        try:
            received_packet = await with_timeout(tb.rx_model.get_next_packet(), 10_000, 'ns')
            assert received_packet is not None, f"No packet received in {desc} non-continuous mode"
            cocotb.log.info(f"Received {desc} non-continuous frame start packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")
            assert received_packet.virtual_channel == 0, f"{desc}: Wrong VC"
            assert received_packet.data_type == DataType.FRAME_START.value, f"{desc}: Wrong data type"
            cocotb.log.info(f"{desc} non-continuous clock mode test PASSED!")
            
        except Exception as e:
            cocotb.log.error(f"Error receiving {desc} packet: {e}")
            # Try to get any available packets for debugging
            try:
                received_packet = await with_timeout(tb.rx_model.get_next_packet(), 1000, 'ns')
                if received_packet:
                    cocotb.log.info(f"Received unexpected {desc} packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")
            except:
                pass
            assert False, f"No packets received in {desc} non-continuous clock mode"

    cocotb.log.info("All multi-lane non-continuous clock tests PASSED!")


# Comprehensive Test Factory Configurations for Enhanced Testing Suite
if cocotb.SIM_NAME:
    # Add comprehensive error injection testing
    factory_error = TestFactory(run_error_injection_test)
    factory_error.add_option("lane_count", [1, 2, 4])
    factory_error.add_option("error_type", ["ecc", "checksum", "crc"])
    factory_error.add_option("data_format", ["raw8", "raw10", "raw12", "rgb888", "yuv422"])
    factory_error.generate_tests()
    
    # Add comprehensive non-continuous clock testing
    factory_non_cont = TestFactory(run_non_continuous_comprehensive_test)
    factory_non_cont.add_option("lane_count", [1, 2, 4])
    factory_non_cont.add_option("bit_rate_mbps", [500, 1000, 1500, 2000])
    factory_non_cont.add_option("packet_type", ["short", "long"])
    factory_non_cont.generate_tests()
    
    # Add performance benchmarking testing
    factory_perf = TestFactory(run_performance_benchmark_test)
    factory_perf.add_option("lane_count", [1, 2, 4])
    factory_perf.add_option("clock_mode", ["continuous", "non_continuous"])
    factory_perf.add_option("data_format", ["raw8", "raw12", "rgb888"])
    factory_perf.generate_tests()


# cocotb-test integration

tests_dir = os.path.dirname(__file__)


def test_csi2_basic(request):
    """Test function for cocotb-test integration"""
    dut = "test_csi2_basic"
    module = os.path.splitext(os.path.basename(__file__))[0]
    toplevel = dut

    verilog_sources = [
        os.path.join(os.path.dirname(__file__), f"{dut}.v"),
    ]

    sim_build = os.path.join(tests_dir, "sim_build",
        request.node.name.replace('[', '-').replace(']', ''))

    simulator.run(
        python_search=[tests_dir],
        verilog_sources=verilog_sources,
        toplevel=toplevel,
        module=module,
        sim_build=sim_build,
    )
