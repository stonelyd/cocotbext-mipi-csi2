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
        assert received_packet.header.validate_ecc(), "Received packet ECC validation failed"
        assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"

        cocotb.log.info(f"Received packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")

    except cocotb.result.SimTimeoutError:
        cocotb.log.warning("Timeout waiting for packet reception")
        raise

    cocotb.log.info("Basic packet transmission test passed")

    # Clean up any incomplete frame state
    await tb.rx_model.reset()

@cocotb.test()
async def test_4lane_packet_transmission(dut):
    """Test CSI-2 packet transmission and reception with 4-lane distribution enabled"""
    setup_logging()
    tb = TB(dut)
    await tb.setup()

    # Configure with 4-lane distribution enabled
    await tb.configure_csi2(lane_count=4, bit_rate_mbps=1000)

    # Override configuration to enable lane distribution
    tb.config.lane_distribution_enabled = True
    tb.tx_phy_model.config.lane_distribution_enabled = True
    tb.rx_phy_model.config.lane_distribution_enabled = True
    cocotb.log.info("4-lane distribution enabled for this test")

    # Reset RX model to ensure clean state
    await tb.rx_model.reset()

    # Test direct PHY transmission with timeout
    frame_start = Csi2ShortPacket.frame_start(virtual_channel=0, frame_number=1)
    packet_bytes = frame_start.to_bytes()

    cocotb.log.info(f"Testing 4-lane PHY transmission: {len(packet_bytes)} bytes")
    cocotb.log.info(f"Packet bytes: {[f'{b:02x}' for b in packet_bytes]}")

    # Send directly via TX PHY with timeout
    try:
        cocotb.log.info("Attempting to start 4-lane packet transmission")
        await with_timeout(tb.tx_phy_model.start_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info("4-lane packet transmission started")
        cocotb.log.info("Attempting to send packet data across 4 lanes")
        await with_timeout(tb.tx_phy_model.send_packet_data(packet_bytes), 100_000_000, 'ns')
        cocotb.log.info("4-lane packet data sent")
        cocotb.log.info("Attempting to stop 4-lane packet transmission")
        await with_timeout(tb.tx_phy_model.stop_packet_transmission(), 100_000_000, 'ns')
        cocotb.log.info("4-lane PHY transmission completed")
    except cocotb.result.SimTimeoutError:
        cocotb.log.error("Timeout in 4-lane PHY transmission")
        raise

    # Wait a bit to ensure transmission is complete
    await Timer(1000, units="ns")

    # Debug: check statistics
    rx_stats = tb.rx_model.get_statistics()
    cocotb.log.info(f"RX stats: {rx_stats}")

    # Wait for RX model to receive the packet
    try:
        received_packet = await tb.rx_model.get_next_packet(timeout_ns=10000)

        assert received_packet is not None, "No packet received"
        assert isinstance(received_packet, Csi2ShortPacket), "Expected short packet"
        assert received_packet.header.validate_ecc(), "Received packet ECC validation failed"
        assert received_packet.data_type == DataType.FRAME_START.value, "Expected frame start packet"
        assert received_packet.virtual_channel == 0, "Expected VC=0"

        cocotb.log.info(f"Received packet: VC={received_packet.virtual_channel}, DT=0x{received_packet.data_type:02x}")

    except cocotb.result.SimTimeoutError:
        cocotb.log.warning("Timeout waiting for packet reception")
        raise


    # Clean up any incomplete frame state
    await tb.rx_model.reset()
