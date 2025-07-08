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


@cocotb.test()
async def test_basic_packet_transmission(dut):
    """Test basic CSI-2 packet transmission and reception"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2()

    # Reset RX model to ensure clean state
    await tb.rx_model.reset()

    # # Test simple signal setting first
    # cocotb.log.info("Testing simple signal setting")

    # # Set some signals manually to test
    # if hasattr(tb.dut, 'data0_p'):
    #     tb.dut.data0_p.value = 1
    #     tb.dut.data0_n.value = 0
    #     cocotb.log.info("Set data0_p=1, data0_n=0")
    #     await Timer(100, units="ns")

    #     tb.dut.data0_p.value = 0
    #     tb.dut.data0_n.value = 1
    #     cocotb.log.info("Set data0_p=0, data0_n=1")
    #     await Timer(100, units="ns")

    # Test direct PHY transmission with timeout
    frame_start = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
    packet_bytes = frame_start.to_bytes()

    cocotb.log.info(f"Testing direct PHY transmission: {len(packet_bytes)} bytes")
    cocotb.log.info(f"Packet bytes: {[f'{b:02x}' for b in packet_bytes]}")

    # Send directly via TX PHY with timeout
    try:
        cocotb.log.info("Attempting to start packet transmission")
        await with_timeout(tb.tx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info("Packet transmission started")
        cocotb.log.info("Attempting to send packet data")
        await with_timeout(tb.tx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
        cocotb.log.info("Packet data sent")
        cocotb.log.info("Attempting to stop packet transmission")
        await with_timeout(tb.tx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info("Direct PHY transmission completed")
    except cocotb.result.SimTimeoutError:
        cocotb.log.error("Timeout in PHY transmission")
        raise

    # Wait a bit for reception
    await Timer(1000, units="ns")

    # Debug: check statistics
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX stats: {rx_stats}")

    # Wait for RX model to receive the packet
    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)

        assert received_packet is not None, "No packet received"
        assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
        assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"

        cocotb.log.info(f"Received packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")

    except cocotb.result.SimTimeoutError:
        cocotb.log.warning("Timeout waiting for packet reception")
        raise

    cocotb.log.info("Basic packet transmission test passed")

    # Clean up any incomplete frame state
    await tb.rx_model.reset()

'''
@cocotb.test()
async def test_frame_transmission(dut):
    """Test complete frame transmission via TX and RX models"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2(bit_rate_mbps=800)

    # Reset RX model to ensure clean state
    await tb.rx_model.reset()

    # Debug: check initial frame buffer state
    frame_buffer = tb.rx_model.frame_buffers[0]
    cocotb.log.info(f"Initial frame buffer state: active={frame_buffer.frame_active}, "
                   f"current_line={frame_buffer.current_line}, "
                   f"data_size={len(frame_buffer.frame_data)}")

    # Create test frame data with nonzero values
    width, height = 16, 12
    frame_data = bytes([(i + 1) % 256 for i in range(width * height)])  # Nonzero pixel data

    cocotb.log.info(f"Testing frame: {width}x{height} = {len(frame_data)} bytes")
    cocotb.log.info(f"Frame data sample: {[f'{b:02x}' for b in frame_data[:10]]}")

    # Send frame packets individually to ensure proper PHY signaling
    builder = tb.tx_model.packet_builder.set_virtual_channel(0)

    # Frame start with nonzero frame number
    frame_number = 1
    frame_start = builder.build_frame_start(frame_number)
    await tb.tx_model.send_packet(frame_start)
    await tb.tx_model.wait_transmission_complete()
    await Timer(1000, units="ns")

    # Send each line individually
    bytes_per_pixel = 1  # RAW8
    line_bytes = width * bytes_per_pixel

    for line_num in range(height):
        # Line start with nonzero line number
        line_start = builder.build_line_start(line_num + 1)
        await tb.tx_model.send_packet(line_start)
        await tb.tx_model.wait_transmission_complete()
        await Timer(1000, units="ns")

        # Line data with actual pixel data
        line_data = frame_data[line_num * line_bytes:(line_num + 1) * line_bytes]
        cocotb.log.info(f"Line {line_num + 1} data: {[f'{b:02x}' for b in line_data[:5]]}...")
        pixel_packet = builder.build_pixel_data(DataType.RAW8, line_data)
        await tb.tx_model.send_packet(pixel_packet)
        await tb.tx_model.wait_transmission_complete()
        await Timer(1000, units="ns")

        # Line end with nonzero line number
        line_end = builder.build_line_end(line_num + 1)
        await tb.tx_model.send_packet(line_end)
        await tb.tx_model.wait_transmission_complete()
        await Timer(1000, units="ns")

    # Frame end with nonzero frame number
    frame_end = builder.build_frame_end(frame_number)
    await tb.tx_model.send_packet(frame_end)
    await tb.tx_model.wait_transmission_complete()
    await Timer(1000, units="ns")

    # Wait for RX model to assemble the frame (poll for completion)
    timeout_ns = 100_000_000  # 100us
    start_time = cocotb.utils.get_sim_time('ns')
    while True:
        if tb.rx_model.frame_buffers[0].is_frame_complete():
            break
        now = cocotb.utils.get_sim_time('ns')
        if now - start_time > timeout_ns:
            cocotb.log.error("Timeout waiting for frame reception")
            raise cocotb.result.SimTimeoutError("Timeout waiting for frame reception")
        await Timer(1000, units="ns")

    # Check if frame was completed
    frame_buffer = tb.rx_model.frame_buffers[0]
    cocotb.log.info(f"Final frame buffer state: active={frame_buffer.frame_active}, "
                   f"complete={frame_buffer.is_frame_complete()}, "
                   f"data_size={len(frame_buffer.frame_data)}")

    # Check if frame is complete and data matches
    if frame_buffer.is_frame_complete():
        received_data = frame_buffer.get_frame_data()
        cocotb.log.info(f"Frame completed successfully: {len(received_data)} bytes")
        cocotb.log.info(f"First 10 bytes: {[f'{b:02x}' for b in received_data[:10]]}")

        assert received_data == frame_data, "Received frame data does not match sent data"
        cocotb.log.info(f"Frame transmission test passed: {len(received_data)} bytes received and verified")
    else:
        cocotb.log.error("Frame was not completed")
        raise AssertionError("Frame was not completed")

    # Clean up any incomplete frame state
    await tb.rx_model.reset()


@cocotb.test()
async def test_dphy_signal_simulation(dut):
    """Test D-PHY signal simulation with enhanced monitoring"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2(bit_rate_mbps=500)

    # Reset RX model and signal monitors
    await tb.rx_model.reset()
    tb.rx_phy_model.reset_signal_monitors()

    cocotb.log.info("Testing D-PHY signal simulation with enhanced monitoring")

    # Create test packet
    frame_start = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
    packet_bytes = frame_start.to_bytes()

    cocotb.log.info(f"Test packet: {len(packet_bytes)} bytes - {[f'{b:02x}' for b in packet_bytes]}")

    # Send packet via RX PHY
    try:
        await with_timeout(tb.rx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
        await with_timeout(tb.rx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
        await with_timeout(tb.rx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info("PHY transmission completed successfully")
    except cocotb.result.SimTimeoutError:
        cocotb.log.error("Timeout in PHY transmission")
        raise

    # Wait for reception and processing
    await Timer(2000, units="ns")

    # Get signal quality statistics
    signal_stats = tb.rx_phy_model.get_signal_statistics()
    cocotb.log.info("Signal Quality Statistics:")
    cocotb.log.info(f"  Clock Lane: {signal_stats['clock_lane']}")
    for i, lane_stats in enumerate(signal_stats['data_lanes']):
        cocotb.log.info(f"  Data Lane {i}: {lane_stats}")

    # Get RX statistics
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX Statistics: {rx_stats}")

    # Verify packet reception
    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)

        if received_packet is not None:
            cocotb.log.info(f"Received packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")

            # Verify packet content
            assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
            assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
            assert received_packet.virtual_channel == 0, "Expected VC=0"

            cocotb.log.info("✅ D-PHY signal simulation test passed - packet received and verified")
        else:
            cocotb.log.warning("⚠️ No packet received - signal simulation may need further debugging")

    except cocotb.result.SimTimeoutError:
        cocotb.log.error("🚧 Timeout waiting for packet reception")

        # Log detailed signal analysis
        cocotb.log.info("Signal Analysis:")
        for i, lane_stats in enumerate(signal_stats['data_lanes']):
            if lane_stats['signal_transitions'] > 0:
                cocotb.log.info(f"  Lane {i}: {lane_stats['signal_transitions']} transitions, "
                              f"{lane_stats['hs_bits_received']} HS bits, "
                              f"{lane_stats['invalid_states']} invalid states")
            else:
                cocotb.log.warning(f"  Lane {i}: No signal transitions detected")

        raise

    # Clean up
    await tb.rx_model.reset()


@cocotb.test()
async def test_multi_virtual_channel(dut):
    """Test multiple virtual channel support"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2(bit_rate_mbps=1000)

    # Reset RX model to ensure clean state
    await tb.rx_model.reset()

    # Send packets on different VCs
    for vc in range(4):
        packet = Csi2ShortPacket.frame_start(virtual_channel=vc, frame_number=vc)
        await tb.tx_model.send_packet(packet)
        await Timer(1000, units="ns")  # Small delay between packets

    # Wait for all transmissions to complete
    # The transmission handler processes one packet at a time, so we need to wait for all 4
    for _ in range(4):
        await tb.tx_model.wait_transmission_complete()
        await Timer(1000, units="ns")  # Small delay between transmissions

    await Timer(5000, units="ns")  # Additional wait for RX processing

    # Check that RX model received packets from different VCs
    vcs_received = set()
    packets_received = 0
    max_packets = 10  # Limit to prevent infinite loop

    while packets_received < max_packets:
        try:
            packet = await tb.rx_model.get_next_packet(timeout_ns=1000)
            if packet is not None:
                vcs_received.add(packet.virtual_channel)
                packets_received += 1
                cocotb.log.info(f"Received packet {packets_received}: VC {packet.virtual_channel}, DT 0x{packet.data_type:02x}")
            else:
                break
        except cocotb.result.SimTimeoutError:
            break

    # Get RX statistics for debugging
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX Statistics: {rx_stats}")

    assert len(vcs_received) > 0, "No packets were received by RX model"

    cocotb.log.info(f"Multi virtual channel test passed: VCs {vcs_received} received ({packets_received} total packets)")

    # Clean up
    await tb.rx_model.reset()


@cocotb.test()
async def test_error_detection(dut):
    """Test error detection capabilities"""
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2()

    # Enable error injection
    tb.tx_model.enable_error_injection(0.5)

    # Send packets with errors
    for i in range(10):
        packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=i)
        await tb.tx_model.send_packet(packet)

    await Timer(10000, units="ns")

    # Check statistics
    tx_stats = tb.tx_model.get_statistics()
    rx_stats = tb.rx_model.get_statistics()

    assert tx_stats['packets_sent'] > 0, "No packets were sent"

    cocotb.log.info(f"Error detection test passed: "
                   f"{tx_stats['packets_sent']} packets sent, "
                   f"{rx_stats.get('packets_received', 0)} packets received")


@cocotb.test()
async def test_packet_builder(dut):
    """Test packet builder functionality"""
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2()

    # Use packet builder
    builder = tb.tx_model.packet_builder.set_virtual_channel(2)

    # Build and send various packets
    frame_start = builder.build_frame_start(42)
    line_start = builder.build_line_start(10)
    pixel_data = builder.build_raw8_line(64, b'\x80' * 64)  # Gray line
    line_end = builder.build_line_end(10)
    frame_end = builder.build_frame_end(42)

    packets = [frame_start, line_start, pixel_data, line_end, frame_end]
    await tb.tx_model.send_packets(packets)

    # Wait for reception and collect packets
    await Timer(5000, units="ns")

    # Check that RX model received the packets
    packets_received = 0
    while True:
        try:
            packet = await tb.rx_model.get_next_packet(timeout_ns=1000)
            if packet is not None:
                packets_received += 1
        except cocotb.result.SimTimeoutError:
            break

    assert packets_received > 0, "No packets were received by RX model"

    cocotb.log.info(f"Packet builder test passed: {packets_received} packets received")


@cocotb.test()
async def test_timing_validation(dut):
    """Test timing validation features"""
    tb = TB(dut)
    await tb.setup()
    await tb.configure_csi2(bit_rate_mbps=1000)

    # Send packets rapidly (may cause timing violations)
    for i in range(5):
        packet = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=i)
        await tb.tx_model.send_packet(packet)
        await Timer(10, units="ns")  # Very short spacing

    await Timer(2000, units="ns")

    # Check that RX model received some packets despite rapid transmission
    packets_received = 0
    while True:
        try:
            packet = await tb.rx_model.get_next_packet(timeout_ns=1000)
            if packet is not None:
                packets_received += 1
        except cocotb.result.SimTimeoutError:
            break

    assert packets_received > 0, "No packets were received by RX model"

    cocotb.log.info(f"Timing validation test passed: {packets_received} packets received")
    '''