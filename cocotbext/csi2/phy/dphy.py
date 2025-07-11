"""
MIPI D-PHY Physical Layer Model

Copyright (c) 2024 CSI-2 Extension Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cocotb
from cocotb.triggers import Timer, Edge
import asyncio
from typing import List
import logging
from enum import Enum
from ..config import Csi2Config, Csi2PhyConfig
from ..bus import Csi2DPhyBus
from ..exceptions import Csi2PhyError, Csi2TimingError
from ..packet import Csi2PacketHeader, DataType


def _is_short_packet_type(data_type: int) -> bool:
    """Check if data type indicates short packet. Based on MIPI CSI-2 Spec."""
    # Generic short packets (Table 12 in CSI-2 v4.0.1)
    if 0x08 <= data_type <= 0x0F:
        return True
    # Synchronization short packets (Table 11 in CSI-2 v4.0.1)
    short_sync_types = {
        DataType.FRAME_START.value,
        DataType.FRAME_END.value,
        DataType.LINE_START.value,
        DataType.LINE_END.value,
    }
    return data_type in short_sync_types


class DPhyLaneType(Enum):
    CLOCK = "clock"
    DATA = "data"


class DPhyLane:
    def __init__(self, lane_type: DPhyLaneType, index: int = 0):
        self.lane_type = lane_type
        self.index = index  # Only meaningful for DATA lanes

    @property
    def name(self) -> str:
        if self.lane_type == DPhyLaneType.CLOCK:
            return "clock"
        else:
            return f"data{self.index}"

    @property
    def signal_name(self) -> str:
        """Get the signal name for bus access"""
        if self.lane_type == DPhyLaneType.CLOCK:
            return "clk"
        else:
            return f"data{self.index}"


class DPhyState:
    """D-PHY Lane States"""
    LP_00 = 0  # Low Power 00
    LP_01 = 1  # Low Power 01
    LP_10 = 2  # Low Power 10
    LP_11 = 3  # Low Power 11
    HS_0 = 4   # High Speed 0
    HS_1 = 5   # High Speed 1

    @staticmethod
    def name(state_val):
        try:
            return {
                0: "LP_00",
                1: "LP_01",
                2: "LP_10",
                3: "LP_11",
                4: "HS_0",
                5: "HS_1"
            }[state_val]
        except Exception:
            return f"UNKNOWN({state_val})"


class DPhyTxTransmitter:
    """D-PHY Transmitter Model"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, lane: DPhyLane):
        """
        Initialize D-PHY transmitter

        Args:
            bus: D-PHY bus interface
            config: CSI-2 configuration
            phy_config: PHY-specific configuration
            lane: Lane object with type and index
        """
        self.bus = bus
        self.config = config
        self.phy_config = phy_config
        self.lane = lane

        # Internal state
        self.current_state = DPhyState.LP_11
        self.hs_active = False
        self.lp_active = True

        # Signal handles - clock lane ignores index
        if lane.lane_type == DPhyLaneType.CLOCK:
            self.sig_p = getattr(bus, 'clk_p', None)
            self.sig_n = getattr(bus, 'clk_n', None)
        else:
            # Data lanes use their index
            self.sig_p = getattr(bus, f'data{lane.index}_p', None)
            self.sig_n = getattr(bus, f'data{lane.index}_n', None)

        if not self.sig_p or not self.sig_n:
            raise Csi2PhyError(f"Missing D-PHY signals for {lane.name}")

        # Debug: log signal handles
        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.tx.{lane.name}')
        self.logger.info(f"Lane {lane.name}: TX signal handles - p: {self.sig_p.value}, n: {self.sig_n.value}")

        # Timing
        self.bit_period_ns = config.get_bit_period_ns()
        self.byte_period_ns = config.get_byte_period_ns()

        # For DDR mode, we need to send at half the bit period
        self.ddr_bit_period_ns = self.bit_period_ns / 2

        # Initialize signals to LP-11 state
        self._set_lp_state(DPhyState.LP_11)
        self.logger.info(f"Lane {lane.name}: TX signals initialized to LP-11 state")

    def _set_lp_state(self, state: int):
        """Set Low Power state on differential signals"""
        self.current_state = state

        if state == DPhyState.LP_00:
            self.sig_p.value = 0
            self.sig_n.value = 0
        elif state == DPhyState.LP_01:
            self.sig_p.value = 0
            self.sig_n.value = 1
        elif state == DPhyState.LP_10:
            self.sig_p.value = 1
            self.sig_n.value = 0
        elif state == DPhyState.LP_11:
            self.sig_p.value = 1
            self.sig_n.value = 1

    async def _send_sync_sequence(self):
        """Send HS Sync-Sequence '00011101' (0xB8 in LSB first) according to D-PHY v2.5 Section 6.4.2"""
        sync_byte = 0xB8  # Binary: 00011101 in LSB first order
        self.logger.info(f"Lane {self.lane.name}: Sending HS Sync-Sequence 0x{sync_byte:02x} (00011101 LSB first)")

        # Send the sync sequence bit by bit
        await self._send_hs_byte(sync_byte)

        self.logger.info(f"Lane {self.lane.name}: HS Sync-Sequence transmission complete")

    def _hs_prepare_sequence_step1_lp01(self):
        """Step 1: Set LP-01 state (no timing)"""
        self._set_lp_state(DPhyState.LP_01)
        self.logger.info(f"Lane {self.lane.name}: Set LP-01 state")

    def _hs_prepare_sequence_step2_lp00(self):
        """Step 2: Set LP-00 state (no timing)"""
        self._set_lp_state(DPhyState.LP_00)
        self.logger.info(f"Lane {self.lane.name}: Set LP-00 state")

    def _hs_prepare_sequence_step3_hs0(self):
        """Step 3: Set HS-0 state (no timing)"""
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0
        self.logger.info(f"Lane {self.lane.name}: Set HS-0 state")

    async def _hs_prepare_sequence_step4_sync(self):
        """Step 4: Send sync sequence (this still needs async for bit-level timing)"""
        await self._send_sync_sequence()
        self.hs_active = True
        self.lp_active = False
        self.logger.info(f"Lane {self.lane.name}: HS prepare sequence complete, ready for data")

    def _hs_exit_sequence_step1_hs0(self):
        """Step 1: Set HS-0 state for trail period (no timing)"""
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0
        self.logger.debug(f"Lane {self.lane.name}: Set HS-0 state for trail")

    def _hs_exit_sequence_step2_lp11(self):
        """Step 2: Set LP-11 state (no timing)"""
        self._set_lp_state(DPhyState.LP_11)
        self.hs_active = False
        self.lp_active = True
        self.logger.debug(f"Lane {self.lane.name}: Set LP-11 state for exit")

    async def send_hs_data(self, data: bytes):
        """
        Send HS data on this lane

        Args:
            data: Data bytes to send
        """
        if not self.hs_active:
            raise Csi2PhyError(f"Lane {self.lane.name}: Cannot send HS data - not in HS mode")

        self.logger.info(f"Lane {self.lane.name}: Sending {len(data)} bytes of HS data")

        for byte_val in data:
            await self._send_hs_byte(byte_val)

        self.logger.info(f"Lane {self.lane.name}: HS data transmission complete")

    async def _send_hs_byte(self, byte_val: int):
        """Send a single byte in HS mode at DDR rate"""
        # Send bits LSB first at DDR rate (half the bit period)
        for bit_pos in range(8):
            bit_val = (byte_val >> bit_pos) & 1

            if bit_val:
                # HS-1: p=1, n=0
                self.sig_p.value = 1
                self.sig_n.value = 0
            else:
                # HS-0: p=0, n=1
                self.sig_p.value = 0
                self.sig_n.value = 1

            # Use DDR timing (half the bit period)
            await Timer(int(self.ddr_bit_period_ns), units='ns')

    async def start_hs_transmission(self):
        """Start HS transmission on this lane"""
        await self._hs_prepare_sequence()

    async def stop_hs_transmission(self):
        """Stop HS transmission on this lane"""
        await self._hs_exit_sequence()

    async def _hs_prepare_sequence(self):
        """Execute HS prepare sequence"""
        self.logger.debug(f"Lane {self.lane.name}: Starting HS prepare")

        # LP-01 state
        self._set_lp_state(DPhyState.LP_01)
        await Timer(50, units='ns')  # LP-01 duration

        # LP-00 state
        self._set_lp_state(DPhyState.LP_00)
        await Timer(60, units='ns')  # LP-00 duration

        # HS prepare
        await Timer(self.phy_config.t_hs_prepare, units='ns')

        # HS-0 state
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0

        # HS zero
        await Timer(self.phy_config.t_hs_zero, units='ns')

        # Send sync sequence
        await self._send_sync_sequence()
        self.hs_active = True
        self.lp_active = False

    async def _hs_exit_sequence(self):
        """Execute HS exit sequence"""
        self.logger.debug(f"Lane {self.lane.name}: Starting HS exit")

        # Drive HS-0 state during HS trail period as per D-PHY spec
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0

        # HS trail
        await Timer(self.phy_config.t_hs_trail, units='ns')

        # LP-11 (HS exit)
        self._set_lp_state(DPhyState.LP_11)
        self.hs_active = False
        self.lp_active = True
        await Timer(self.phy_config.t_hs_exit, units='ns')


class DPhyTxModel:
    """D-PHY Transmitter Model for multi-lane PHY"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config):
        """
        Initialize D-PHY transmitter model

        Args:
            bus: D-PHY bus interface
            config: CSI-2 configuration
        """
        self.bus = bus
        self.config = config

        # Create PHY config with timing from CSI-2 config
        self.phy_config = Csi2PhyConfig(
            t_clk_prepare=config.t_clk_prepare_ns,
            t_clk_zero=config.t_clk_zero_ns,
            t_clk_pre=config.t_clk_pre_ns,
            t_clk_post=config.t_clk_post_ns,
            t_clk_trail=config.t_clk_trail_ns,
            t_hs_prepare=config.t_hs_prepare_ns,
            t_hs_zero=config.t_hs_zero_ns,
            t_hs_trail=config.t_hs_trail_ns,
            t_hs_exit=config.t_hs_exit_ns
        )

        # Validate timing parameters
        if not self.phy_config.validate_timing(config.bit_rate_mbps):
            # Debug output to see what's failing
            ui_ns = 1000.0 / config.bit_rate_mbps
            print(f"Timing validation failed for {config.bit_rate_mbps} Mbps:")
            print(f"  UI = {ui_ns} ns")
            print(f"  t_clk_prepare = {self.phy_config.t_clk_prepare} ns (min: 38.0)")
            print(f"  t_hs_prepare = {self.phy_config.t_hs_prepare} ns (min: {40.0 + 4 * ui_ns})")
            print(f"  t_hs_zero = {self.phy_config.t_hs_zero} ns (min: {105.0 + 6 * ui_ns})")
            raise Csi2TimingError("Invalid timing parameters for bit rate")

        # Create lanes with clear identification
        self.clock_lane = DPhyLane(DPhyLaneType.CLOCK)  # index ignored
        self.data_lanes = [DPhyLane(DPhyLaneType.DATA, i) for i in range(config.lane_count)]

        # Create transmitters
        self.clock_tx = DPhyTxTransmitter(bus, config, self.phy_config, self.clock_lane)
        self.data_tx = [DPhyTxTransmitter(bus, config, self.phy_config, lane) for lane in self.data_lanes]

        # For backward compatibility, create tx_lanes list (only data lanes)
        self.tx_lanes = self.data_tx

        self.logger = logging.getLogger('cocotbext.csi2.dphy.tx_model')

        # Clock generation for continuous clock mode
        if config.continuous_clock:
            self._clock_task = cocotb.start_soon(self._generate_continuous_clock())

        # Monitoring state
        self.transmission_active = False
        self.packet_count = 0
        self.bytes_transmitted = 0

    async def _generate_continuous_clock(self):
        """Generate continuous HS clock"""
        period_ns = self.config.get_bit_period_ns()

        while True:
            # Generate differential clock
            self.clock_tx.sig_p.value = 1
            self.clock_tx.sig_n.value = 0
            await Timer(period_ns / 2, units='ns')

            self.clock_tx.sig_p.value = 0
            self.clock_tx.sig_n.value = 1
            await Timer(period_ns / 2, units='ns')

    async def send_packet_data(self, data: bytes):
        """
        Send packet data across all data lanes

        Args:
            data: Packet data to transmit
        """
        self.logger.info(f"TX PHY: send_packet_data called with {len(data)} bytes: {[f'0x{b:02x}' for b in data]}")

        if not self.data_tx:
            raise Csi2PhyError("No data lanes configured")

        # Start monitoring transmission
        self.transmission_active = True
        self.packet_count += 1
        self.bytes_transmitted += len(data)

        if self.config.lane_distribution_enabled and len(self.data_tx) > 1:
            # Distribute data across lanes
            from ..utils import bytes_to_lanes
            lane_data = bytes_to_lanes(data, len(self.data_tx))

            # Send data on all lanes sequentially (cocotb compatibility)
            for tx_lane, lane_bytes in zip(self.data_tx, lane_data):
                self.logger.info(f"TX PHY: Sending {len(lane_bytes)} bytes to {tx_lane.lane.name}")
                await tx_lane.send_hs_data(lane_bytes)
        else:
            # Send all data on the first lane (no distribution)
            self.logger.info(f"TX PHY: Sending {len(data)} bytes to {self.data_tx[0].lane.name}")
            await self.data_tx[0].send_hs_data(data)

        # End monitoring transmission
        self.transmission_active = False

    async def start_packet_transmission(self):
        """Start packet transmission (HS prepare on all lanes) with parallel timing"""
        self.logger.info("TX PHY: Starting packet transmission with parallel timing")

        if not self.config.continuous_clock:
            await self.clock_tx.start_hs_transmission()

        # Step 1: Set LP-01 state on all lanes simultaneously
        for tx_lane in self.data_tx:
            tx_lane._hs_prepare_sequence_step1_lp01()

        # Wait for LP-01 duration (50ns)
        await Timer(50, units='ns')
        self.logger.info("TX PHY: LP-01 duration complete (50ns)")

        # Step 2: Set LP-00 state on all lanes simultaneously
        for tx_lane in self.data_tx:
            tx_lane._hs_prepare_sequence_step2_lp00()

        # Wait for LP-00 duration (60ns)
        await Timer(60, units='ns')
        self.logger.info("TX PHY: LP-00 duration complete (60ns)")

        # Wait for HS prepare time
        await Timer(int(self.phy_config.t_hs_prepare), units='ns')
        self.logger.info(f"TX PHY: HS prepare duration complete ({self.phy_config.t_hs_prepare}ns)")

        # Step 3: Set HS-0 state on all lanes simultaneously
        for tx_lane in self.data_tx:
            tx_lane._hs_prepare_sequence_step3_hs0()

        # Wait for HS zero period
        await Timer(int(self.phy_config.t_hs_zero), units='ns')
        self.logger.info(f"TX PHY: HS zero duration complete ({self.phy_config.t_hs_zero}ns)")

        # Step 4: Send sync sequence on all lanes (this still needs async for bit-level timing)
        sync_tasks = [cocotb.start_soon(tx_lane._hs_prepare_sequence_step4_sync()) for tx_lane in self.data_tx]
        for task in sync_tasks:
            await task
        self.logger.info("TX PHY: All lanes sync sequences complete")

    async def stop_packet_transmission(self):
        """Stop packet transmission (HS exit on all lanes) with parallel timing"""
        self.logger.info("TX PHY: Stopping packet transmission with parallel timing")

        # Step 1: Set HS-0 state on all lanes simultaneously for trail period
        for tx_lane in self.data_tx:
            tx_lane._hs_exit_sequence_step1_hs0()

        # Wait for HS trail period
        await Timer(int(self.phy_config.t_hs_trail), units='ns')
        self.logger.info(f"TX PHY: HS trail duration complete ({self.phy_config.t_hs_trail}ns)")

        # Step 2: Set LP-11 state on all lanes simultaneously
        for tx_lane in self.data_tx:
            tx_lane._hs_exit_sequence_step2_lp11()

        # Wait for HS exit period
        await Timer(int(self.phy_config.t_hs_exit), units='ns')
        self.logger.info(f"TX PHY: HS exit duration complete ({self.phy_config.t_hs_exit}ns)")

        # Handle clock lane if not continuous
        if not self.config.continuous_clock:
            await self.clock_tx.stop_hs_transmission()

    def get_transmission_statistics(self) -> dict:
        """Get transmission statistics"""
        return {
            'packet_count': self.packet_count,
            'bytes_transmitted': self.bytes_transmitted,
            'transmission_active': self.transmission_active,
            'lane_count': len(self.data_tx),
            'continuous_clock': self.config.continuous_clock
        }

    def reset_transmission_statistics(self):
        """Reset transmission statistics"""
        self.packet_count = 0
        self.bytes_transmitted = 0
        self.transmission_active = False


class DPhyRxModel:
    """D-PHY Receiver Model for multi-lane PHY"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config):
        """
        Initialize D-PHY receiver model

        Args:
            bus: D-PHY bus interface
            config: CSI-2 configuration
        """
        self.bus = bus
        self.config = config

        # Create PHY config with timing from CSI-2 config
        self.phy_config = Csi2PhyConfig(
            t_clk_prepare=config.t_clk_prepare_ns,
            t_clk_zero=config.t_clk_zero_ns,
            t_clk_pre=config.t_clk_pre_ns,
            t_clk_post=config.t_clk_post_ns,
            t_clk_trail=config.t_clk_trail_ns,
            t_hs_prepare=config.t_hs_prepare_ns,
            t_hs_zero=config.t_hs_zero_ns,
            t_hs_trail=config.t_hs_trail_ns,
            t_hs_exit=config.t_hs_exit_ns
        )
        self.logger = logging.getLogger('cocotbext.csi2.dphy.rx_model')

        # Create receivers for clock and data lanes
        self.clock_lane = DPhyLane(DPhyLaneType.CLOCK)
        self.data_lanes = [DPhyLane(DPhyLaneType.DATA, i) for i in range(config.lane_count)]

        # Callbacks
        self.on_packet_start = None
        self.on_packet_end = None
        self.on_data_received = None

        # Lane-specific buffers and state tracking
        self.lane_buffers = {i: [] for i in range(config.lane_count)}
        self.lane_bit_shifters = {i: 0 for i in range(config.lane_count)}
        self.lane_byte_buffers = {i: 0 for i in range(config.lane_count)}
        self.lane_bit_counts = {i: 0 for i in range(config.lane_count)}
        self.lane_sync_aligned = {i: False for i in range(config.lane_count)}
        self.lane_sync_shift = {i: 0 for i in range(config.lane_count)}
        self.lane_processing_packet = {i: False for i in range(config.lane_count)}
        self.hs_active = False

        # Packet length awareness
        self.lane_packet_lengths = {i: 0 for i in range(config.lane_count)}

        # Sync sequence (SoT) from D-PHY spec v2.5, section 6.4.2
        self.sync_sequence_bits = 0xB8

        # Lane enable status
        self.enabled_lanes = set(range(config.lane_count))

        # Start monitoring tasks
        self._clock_monitor_task = cocotb.start_soon(self._monitor_clock_events())

    async def _monitor_clock_events(self):
        """Monitor clock lane for edge events"""
        while True:
            # D-PHY is DDR, so we must sample on every clock edge
            await Edge(self.bus.clk_p)
            for lane_idx in self.enabled_lanes:
                await self._sample_data_lane(lane_idx)

    async def _sample_data_lane(self, lane_idx: int):
        """Sample a specific data lane."""
        try:
            p_val = int(getattr(self.bus, f'data{lane_idx}_p').value)
            n_val = int(getattr(self.bus, f'data{lane_idx}_n').value)
        except (ValueError, TypeError):
            return

        # HS-0: p=0, n=1; HS-1: p=1, n=0
        if p_val == 0 and n_val == 1:
            await self._handle_data_bit(lane_idx, False)
        elif p_val == 1 and n_val == 0:
            await self._handle_data_bit(lane_idx, True)
        elif p_val == 1 and n_val == 1:  # LP-11, potential end of packet
            await self._handle_packet_end(lane_idx)

    async def _handle_data_bit(self, lane_idx: int, bit_value: bool):
        """Handle received data bit from a specific lane."""
        if not self.lane_sync_aligned[lane_idx]:
            self.lane_bit_shifters[lane_idx] = ((self.lane_bit_shifters[lane_idx] >> 1) | (bit_value << 15)) & 0xFFFF
            for shift in range(9):
                shifted_bits = (self.lane_bit_shifters[lane_idx] >> (8 - shift)) & 0xFF
                if shifted_bits == self.sync_sequence_bits:
                    self.lane_sync_aligned[lane_idx] = True
                    self.lane_sync_shift[lane_idx] = shift
                    self.lane_processing_packet[lane_idx] = True
                    self.logger.info(f"Lane {lane_idx}: Sync detected.")
                    self.lane_byte_buffers[lane_idx] = 0
                    self.lane_bit_counts[lane_idx] = 0

                    is_first_lane_to_sync = not self.hs_active
                    if is_first_lane_to_sync:
                        self.hs_active = True
                        if self.on_packet_start:
                            await cocotb.start(self.on_packet_start())
                    return

        if self.lane_sync_aligned[lane_idx]:
            if bit_value:
                self.lane_byte_buffers[lane_idx] |= (1 << self.lane_bit_counts[lane_idx])
            self.lane_bit_counts[lane_idx] += 1
            if self.lane_bit_counts[lane_idx] >= 8:
                byte_val = self.lane_byte_buffers[lane_idx]
                await self._process_byte(lane_idx, byte_val)
                self.lane_byte_buffers[lane_idx] = 0
                self.lane_bit_counts[lane_idx] = 0

    async def _process_byte(self, lane_idx: int, byte_val: int):
        """Process a fully assembled byte, with packet-awareness."""
        packet_len = self.lane_packet_lengths.get(lane_idx, 0)
        bytes_recvd = len(self.lane_buffers[lane_idx])

        # If we already know the length and have received it all, do nothing.
        if packet_len > 0 and bytes_recvd >= packet_len:
            self.logger.debug(f"Lane {lane_idx}: Ignoring trailing byte, packet complete.")
            return

        # Buffer and forward the valid byte
        self.lane_buffers[lane_idx].append(byte_val)
        if self.on_data_received:
            await cocotb.start(self.on_data_received(byte_val))

        # If header is now complete, determine full packet length
        if len(self.lane_buffers[lane_idx]) == 4:
            header_bytes = bytes(self.lane_buffers[lane_idx])
            try:
                header = Csi2PacketHeader.from_bytes(header_bytes)
                if _is_short_packet_type(header.data_type):
                    self.lane_packet_lengths[lane_idx] = 4
                    self.logger.info(f"Lane {lane_idx}: Short packet detected. Expecting 4 bytes total.")
                else:  # Long packet
                    length = 4 + header.word_count + 2
                    self.lane_packet_lengths[lane_idx] = length
                    self.logger.info(f"Lane {lane_idx}: Long packet detected. WC={header.word_count}. Expecting {length} bytes total.")
            except Exception as e:
                self.logger.error(f"Lane {lane_idx}: Header parse failed: {e}. Cannot determine packet length.")
                self.lane_packet_lengths[lane_idx] = 0  # Reset to unknown

    async def _handle_packet_end(self, lane_idx: int):
        """Handle end of packet on a specific lane."""
        if not self.lane_processing_packet.get(lane_idx, False):
            return

        self.logger.info(f"Lane {lane_idx}: Packet end, {len(self.lane_buffers[lane_idx])} bytes received")
        self.reset_lane_state(lane_idx)

        # Check if all lanes are done processing
        if all(not self.lane_processing_packet[i] for i in self.enabled_lanes):
            if self.hs_active:
                self.hs_active = False
                if self.on_packet_end:
                    await cocotb.start(self.on_packet_end())

    def reset_lane_state(self, lane_idx: int):
        """Reset the state for a single lane."""
        self.lane_buffers[lane_idx].clear()
        self.lane_bit_shifters[lane_idx] = 0
        self.lane_sync_aligned[lane_idx] = False
        self.lane_sync_shift[lane_idx] = 0
        self.lane_processing_packet[lane_idx] = False
        self.lane_byte_buffers[lane_idx] = 0
        self.lane_bit_counts[lane_idx] = 0
        self.lane_packet_lengths[lane_idx] = 0

    def reset(self):
        """Reset the entire receiver model state."""
        for i in range(self.config.lane_count):
            self.reset_lane_state(i)
        self.hs_active = False
        self.logger.info("DPhyRxModel reset.")

    def enable_lane(self, lane_idx: int):
        if 0 <= lane_idx < self.config.lane_count:
            self.enabled_lanes.add(lane_idx)
        else:
            self.logger.error(f"Invalid lane index: {lane_idx}")

    def disable_lane(self, lane_idx: int):
        if lane_idx in self.enabled_lanes:
            self.enabled_lanes.remove(lane_idx)

    def get_received_data(self) -> bytes:
        # Note: This assumes single-lane operation for simplicity.
        # Multi-lane would require a more sophisticated byte de-skew and merge algorithm.
        return bytes(self.lane_buffers.get(0, []))

    def set_rx_callbacks(self, on_packet_start=None, on_packet_end=None, on_data_received=None):
        self.on_packet_start = on_packet_start
        self.on_packet_end = on_packet_end
        self.on_data_received = on_data_received
