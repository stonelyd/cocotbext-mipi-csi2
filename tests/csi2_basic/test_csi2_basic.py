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

from pathlib import Path
from cocotb.runner import get_runner


from cocotb.regression import TestFactory

from cocotbext.csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2Bus, Csi2DPhyBus, Csi2ShortPacket, Csi2LongPacket
)
from cocotbext.csi2.utils import setup_logging


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
            from cocotbext.csi2.phy import DPhyTxModel, DPhyRxModel
            self.tx_phy_model = DPhyTxModel(self.bus, self.config)
            self.rx_phy_model = DPhyRxModel(self.bus, self.config)
        else:
            self.bus = Csi2Bus.from_entity(self.dut)
            from cocotbext.csi2.phy import CPhyModel
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

# Consolidated long packet test
async def run_long_packet_transmission(dut, lane_count=4, data_format="raw8", **kwargs):
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
        from cocotbext.csi2.utils import pack_raw10
        payload_data = pack_raw10(pixels)
        data_type = DataType.RAW10
        expected_word_count = 20
        expected_payload_length = 20
        format_name = "Raw10"
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

async def run_frame_transmission(dut, lane_count=4):
    """Test complete 2-lane frame transmission using event-driven RX (no timer-based polling)"""

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

    # Add long packet factory
    factory_long = TestFactory(run_long_packet_transmission)
    factory_long.add_option("lane_count", [1, 2, 4])
    factory_long.add_option("data_format", ["raw8", "raw10"])
    factory_long.generate_tests()

    # Add frame transmission factory
    factory_frame = TestFactory(run_frame_transmission)
    factory_frame.add_option("lane_count", [1, 2, 4])
    factory_frame.generate_tests()







class Test_Csi2Basic:

    tests_dir = os.path.dirname(__file__)




    def test_csi2_basic_runner():
        sim = os.getenv("SIM", "icarus")

        proj_path = Path(__file__).resolve().parent

        dut = "test_csi2_basic"
        sources = [os.path.join(os.path.dirname(__file__), f"{dut}.v"), ]

        runner = get_runner(sim)
        runner.build(
            sources=sources,
            hdl_toplevel="test_csi2_basic",
        )

        runner.test(hdl_toplevel="test_csi2_basic", test_module="test_csi2_basic,")


# # cocotb-test

# tests_dir = os.path.dirname(__file__)


# def test_pcie(request):
#     dut = "test_pcie"
#     module = os.path.splitext(os.path.basename(__file__))[0]
#     toplevel = dut

#     verilog_sources = [
#         os.path.join(os.path.dirname(__file__), f"{dut}.v"),
#     ]

#     sim_build = os.path.join(tests_dir, "sim_build",
#         request.node.name.replace('[', '-').replace(']', ''))

#     cocotb_test.simulator.run(
#         python_search=[tests_dir],
#         verilog_sources=verilog_sources,
#         toplevel=toplevel,
#         module=module,
#         sim_build=sim_build,
#     )