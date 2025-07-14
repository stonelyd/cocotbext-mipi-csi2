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

        # Initialize lane state (previously in DPhyTxTransmitter)
        self.lane_states = {}
        self.lane_hs_active = {}
        self.lane_lp_active = {}
        self.lane_signals = {}
        self.lane_loggers = {}

        # Initialize clock lane
        self._init_lane(self.clock_lane)

        # Initialize data lanes
        for lane in self.data_lanes:
            self._init_lane(lane)

        # For backward compatibility, create tx_lanes list (data lanes for signal access)
        self.tx_lanes = []
        for lane in self.data_lanes:
            lane_obj = type('LaneObj', (), {
                'lane': lane,
                'send_hs_data': lambda data, l=lane: self._send_hs_data_on_lane(l, data),
                '_hs_prepare_sequence_step1_lp01': lambda l=lane: self._hs_prepare_sequence_step1_lp01(l),
                '_hs_prepare_sequence_step2_lp00': lambda l=lane: self._hs_prepare_sequence_step2_lp00(l),
                '_hs_prepare_sequence_step3_hs0': lambda l=lane: self._hs_prepare_sequence_step3_hs0(l),
                '_hs_prepare_sequence_step4_sync': lambda l=lane: self._hs_prepare_sequence_step4_sync(l),
                '_hs_exit_sequence_step1_hs0': lambda l=lane: self._hs_exit_sequence_step1_hs0(l),
                '_hs_exit_sequence_step2_lp11': lambda l=lane: self._hs_exit_sequence_step2_lp11(l),
                'start_hs_transmission': lambda l=lane: self._start_hs_transmission(l),
                'stop_hs_transmission': lambda l=lane: self._stop_hs_transmission(l),
            })()
            self.tx_lanes.append(lane_obj)

        # Create clock transmitter-like object for backward compatibility
        self.clock_tx = type('ClockTx', (), {
            'start_hs_transmission': lambda: self._start_hs_transmission(self.clock_lane),
            'stop_hs_transmission': lambda: self._stop_hs_transmission(self.clock_lane),
            'sig_p': self.lane_signals[self.clock_lane.name]['p'],
            'sig_n': self.lane_signals[self.clock_lane.name]['n'],
        })()

        # For backward compatibility, create data_tx list
        self.data_tx = self.tx_lanes

        self.logger = logging.getLogger('cocotbext.csi2.dphy.tx_model')

        # Timing
        self.bit_period_ns = config.get_bit_period_ns()
        self.byte_period_ns = config.get_byte_period_ns()
        self.ddr_bit_period_ns = self.bit_period_ns / 2

        # Clock generation for continuous clock mode
        if config.continuous_clock:
            self._clock_task = cocotb.start_soon(self._generate_continuous_clock())

        # Monitoring state
        self.transmission_active = False
        self.packet_count = 0
        self.bytes_transmitted = 0

    def _init_lane(self, lane: DPhyLane):
        """Initialize a lane with its state and signal handles"""
        # Internal state
        self.lane_states[lane.name] = DPhyState.LP_11
        self.lane_hs_active[lane.name] = False
        self.lane_lp_active[lane.name] = True

        # Signal handles - clock lane ignores index
        if lane.lane_type == DPhyLaneType.CLOCK:
            sig_p = getattr(self.bus, 'clk_p', None)
            sig_n = getattr(self.bus, 'clk_n', None)
        else:
            # Data lanes use their index
            sig_p = getattr(self.bus, f'data{lane.index}_p', None)
            sig_n = getattr(self.bus, f'data{lane.index}_n', None)

        if not sig_p or not sig_n:
            raise Csi2PhyError(f"Missing D-PHY signals for {lane.name}")

        self.lane_signals[lane.name] = {'p': sig_p, 'n': sig_n}

        # Logger
        self.lane_loggers[lane.name] = logging.getLogger(f'cocotbext.csi2.dphy.tx.{lane.name}')
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: TX signal handles - p: {sig_p.value}, n: {sig_n.value}")

        # Initialize signals to LP-11 state
        self._set_lp_state(lane, DPhyState.LP_11)
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: TX signals initialized to LP-11 state")

    def _set_lp_state(self, lane: DPhyLane, state: int):
        """Set Low Power state on differential signals"""
        lane_name = lane.name
        self.lane_states[lane_name] = state

        signals = self.lane_signals[lane_name]

        if state == DPhyState.LP_00:
            signals['p'].value = 0
            signals['n'].value = 0
        elif state == DPhyState.LP_01:
            signals['p'].value = 0
            signals['n'].value = 1
        elif state == DPhyState.LP_10:
            signals['p'].value = 1
            signals['n'].value = 0
        elif state == DPhyState.LP_11:
            signals['p'].value = 1
            signals['n'].value = 1

    async def _send_sync_sequence(self, lane: DPhyLane):
        """Send HS Sync-Sequence '00011101' (0xB8 in LSB first) according to D-PHY v2.5 Section 6.4.2"""
        sync_byte = 0xB8  # Binary: 00011101 in LSB first order
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: Sending HS Sync-Sequence 0x{sync_byte:02x} (00011101 LSB first)")

        # Send the sync sequence bit by bit
        await self._send_hs_byte(lane, sync_byte)

        self.lane_loggers[lane.name].info(f"Lane {lane.name}: HS Sync-Sequence transmission complete")

    def _hs_prepare_sequence_step1_lp01(self, lane: DPhyLane):
        """Step 1: Set LP-01 state (no timing)"""
        self._set_lp_state(lane, DPhyState.LP_01)
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: Set LP-01 state")

    def _hs_prepare_sequence_step2_lp00(self, lane: DPhyLane):
        """Step 2: Set LP-00 state (no timing)"""
        self._set_lp_state(lane, DPhyState.LP_00)
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: Set LP-00 state")

    def _hs_prepare_sequence_step3_hs0(self, lane: DPhyLane):
        """Step 3: Set HS-0 state (no timing)"""
        signals = self.lane_signals[lane.name]
        signals['p'].value = 0
        signals['n'].value = 1
        self.lane_states[lane.name] = DPhyState.HS_0
        self.lane_loggers[lane.name].info(f"Lane {lane.name}: Set HS-0 state")

    async def _hs_prepare_sequence_step4_sync(self, lane: DPhyLane):
        """Step 4: Send sync sequence (this still needs async for bit-level timing)"""
        # This method is now called for each lane but will be coordinated by the parent
        # The actual sync transmission is handled by _send_sync_sequence_all_lanes
        pass

    async def _send_sync_sequence_all_lanes(self):
        """Send HS Sync-Sequence across all lanes simultaneously"""
        sync_byte = 0xB8  # Binary: 00011101 in LSB first order
        self.logger.info(f"TX PHY: Sending HS Sync-Sequence 0x{sync_byte:02x} (00011101 LSB first) across all lanes")

        # Send the sync sequence bit by bit across all lanes simultaneously
        for bit_pos in range(8):
            bit_val = (sync_byte >> bit_pos) & 1

            # Set the same bit value on all lanes simultaneously
            for lane in self.data_lanes:
                signals = self.lane_signals[lane.name]
                if bit_val:
                    # HS-1: p=1, n=0
                    signals['p'].value = 1
                    signals['n'].value = 0
                else:
                    # HS-0: p=0, n=1
                    signals['p'].value = 0
                    signals['n'].value = 1

            # Use proper DDR timing (half the bit period) - same timing for all lanes
            # For DDR, each bit period is half the UI (Unit Interval)
            timer_value = max(1, int(self.bit_period_ns / 2))
            await Timer(timer_value, units='ns')

        # Mark all lanes as HS active
        for lane in self.data_lanes:
            self.lane_hs_active[lane.name] = True
            self.lane_lp_active[lane.name] = False
            self.lane_loggers[lane.name].info(f"Lane {lane.name}: HS prepare sequence complete, ready for data")

        self.logger.info("TX PHY: All lanes sync sequences complete")

    def _hs_exit_sequence_step1_hs0(self, lane: DPhyLane):
        """Step 1: Set HS-0 state for trail period (no timing)"""
        signals = self.lane_signals[lane.name]
        signals['p'].value = 0
        signals['n'].value = 1
        self.lane_states[lane.name] = DPhyState.HS_0
        self.lane_loggers[lane.name].debug(f"Lane {lane.name}: Set HS-0 state for trail")

    def _hs_exit_sequence_step2_lp11(self, lane: DPhyLane):
        """Step 2: Set LP-11 state (no timing)"""
        self._set_lp_state(lane, DPhyState.LP_11)
        self.lane_hs_active[lane.name] = False
        self.lane_lp_active[lane.name] = True
        self.lane_loggers[lane.name].debug(f"Lane {lane.name}: Set LP-11 state for exit")

    async def _send_hs_data_on_lane(self, lane: DPhyLane, data: bytes):
        """
        Send HS data on a specific lane

        Args:
            lane: Lane to send data on
            data: Data bytes to send
        """
        if not self.lane_hs_active[lane.name]:
            raise Csi2PhyError(f"Lane {lane.name}: Cannot send HS data - not in HS mode")

        self.lane_loggers[lane.name].info(f"Lane {lane.name}: Sending {len(data)} bytes of HS data")

        for byte_val in data:
            await self._send_hs_byte(lane, byte_val)

        self.lane_loggers[lane.name].info(f"Lane {lane.name}: HS data transmission complete")

    async def _send_hs_byte(self, lane: DPhyLane, byte_val: int):
        """Send a single byte in HS mode at DDR rate"""
        signals = self.lane_signals[lane.name]

        # Send bits LSB first at DDR rate (half the bit period)
        for bit_pos in range(8):
            bit_val = (byte_val >> bit_pos) & 1

            if bit_val:
                # HS-1: p=1, n=0
                signals['p'].value = 1
                signals['n'].value = 0
            else:
                # HS-0: p=0, n=1
                signals['p'].value = 0
                signals['n'].value = 1

            # Use DDR timing (half the bit period)
            # Ensure minimum timer resolution of 1ns to avoid undefined behavior
            timer_value = self.ddr_bit_period_ns
            await Timer(timer_value, units='ns')

    async def _start_hs_transmission(self, lane: DPhyLane):
        """Start HS transmission on a specific lane"""
        await self._hs_prepare_sequence(lane)

    async def _stop_hs_transmission(self, lane: DPhyLane):
        """Stop HS transmission on a specific lane"""
        await self._hs_exit_sequence(lane)

    async def _hs_prepare_sequence(self, lane: DPhyLane):
        """Execute HS prepare sequence for a specific lane"""
        self.lane_loggers[lane.name].debug(f"Lane {lane.name}: Starting HS prepare")

        # LP-01 state
        self._set_lp_state(lane, DPhyState.LP_01)
        await Timer(50, units='ns')  # LP-01 duration

        # LP-00 state
        self._set_lp_state(lane, DPhyState.LP_00)
        await Timer(60, units='ns')  # LP-00 duration

        # HS prepare
        await Timer(self.phy_config.t_hs_prepare, units='ns')

        # HS-0 state
        signals = self.lane_signals[lane.name]
        signals['p'].value = 0
        signals['n'].value = 1
        self.lane_states[lane.name] = DPhyState.HS_0

        # HS zero
        await Timer(self.phy_config.t_hs_zero, units='ns')

        # Send sync sequence
        await self._send_sync_sequence(lane)
        self.lane_hs_active[lane.name] = True
        self.lane_lp_active[lane.name] = False

    async def _hs_exit_sequence(self, lane: DPhyLane):
        """Execute HS exit sequence for a specific lane"""
        self.lane_loggers[lane.name].debug(f"Lane {lane.name}: Starting HS exit")

        # Drive HS-0 state during HS trail period as per D-PHY spec
        signals = self.lane_signals[lane.name]
        signals['p'].value = 0
        signals['n'].value = 1
        self.lane_states[lane.name] = DPhyState.HS_0

        # HS trail
        await Timer(self.phy_config.t_hs_trail, units='ns')

        # LP-11 (HS exit)
        self._set_lp_state(lane, DPhyState.LP_11)
        self.lane_hs_active[lane.name] = False
        self.lane_lp_active[lane.name] = True
        await Timer(self.phy_config.t_hs_exit, units='ns')

    async def _generate_continuous_clock(self):
        """Generate continuous HS clock"""
        period_ns = self.config.get_bit_period_ns()
        clock_signals = self.lane_signals[self.clock_lane.name]

        while True:
            # Generate differential clock
            clock_signals['p'].value = 1
            clock_signals['n'].value = 0
            await Timer(period_ns / 2, units='ns')

            clock_signals['p'].value = 0
            clock_signals['n'].value = 1
            await Timer(period_ns / 2, units='ns')

    async def send_packet_data(self, data: bytes):
        """
        Send packet data across all data lanes

        Args:
            data: Packet data to transmit
        """
        self.logger.info(f"TX PHY: send_packet_data called with {len(data)} bytes: {[f'0x{b:02x}' for b in data]}")

        if not self.data_lanes:
            raise Csi2PhyError("No data lanes configured")

        # Start monitoring transmission
        self.transmission_active = True
        self.packet_count += 1
        self.bytes_transmitted += len(data)

        if self.config.lane_distribution_enabled and len(self.data_lanes) > 1:
            # Distribute data across lanes
            from ..utils import bytes_to_lanes
            lane_data = bytes_to_lanes(data, len(self.data_lanes))

            # Send data on all lanes simultaneously (parallel transmission)
            self.logger.info(f"TX PHY: Sending distributed data across {len(self.data_lanes)} lanes")
            for i, (lane, lane_bytes) in enumerate(zip(self.data_lanes, lane_data)):
                self.logger.info(f"TX PHY: Lane {i} will send {len(lane_bytes)} bytes: {[f'0x{b:02x}' for b in lane_bytes]}")

            # Send data byte-by-byte across all lanes simultaneously
            max_bytes = max(len(lane_bytes) for lane_bytes in lane_data)
            for byte_idx in range(max_bytes):
                # Send one byte on each lane simultaneously
                for lane_idx, (lane, lane_bytes) in enumerate(zip(self.data_lanes, lane_data)):
                    if byte_idx < len(lane_bytes):
                        byte_val = lane_bytes[byte_idx]
                        await self._send_hs_byte(lane, byte_val)
                    else:
                        # Pad with zeros if this lane has fewer bytes
                        await self._send_hs_byte(lane, 0)

            self.logger.info("TX PHY: All lanes data transmission complete")
        else:
            # Send all data on the first lane (no distribution)
            self.logger.info(f"TX PHY: Sending {len(data)} bytes to {self.data_lanes[0].name}")
            await self._send_hs_data_on_lane(self.data_lanes[0], data)

        # End monitoring transmission
        self.transmission_active = False

    async def start_packet_transmission(self):
        """Start packet transmission (HS prepare on all lanes) with parallel timing"""
        self.logger.info("TX PHY: Starting packet transmission with parallel timing")

        if not self.config.continuous_clock:
            await self._start_hs_transmission(self.clock_lane)

        # Step 1: Set LP-01 state on all lanes simultaneously
        for lane in self.data_lanes:
            self._hs_prepare_sequence_step1_lp01(lane)

        # Wait for LP-01 duration (50ns)
        await Timer(50, units='ns')
        self.logger.info("TX PHY: LP-01 duration complete (50ns)")

        # Step 2: Set LP-00 state on all lanes simultaneously
        for lane in self.data_lanes:
            self._hs_prepare_sequence_step2_lp00(lane)

        # Wait for LP-00 duration (60ns)
        await Timer(60, units='ns')
        self.logger.info("TX PHY: LP-00 duration complete (60ns)")

        # Wait for HS prepare time
        await Timer(int(self.phy_config.t_hs_prepare), units='ns')
        self.logger.info(f"TX PHY: HS prepare duration complete ({self.phy_config.t_hs_prepare}ns)")

        # Step 3: Set HS-0 state on all lanes simultaneously
        for lane in self.data_lanes:
            self._hs_prepare_sequence_step3_hs0(lane)

        # Wait for HS zero period
        await Timer(int(self.phy_config.t_hs_zero), units='ns')
        self.logger.info(f"TX PHY: HS zero duration complete ({self.phy_config.t_hs_zero}ns)")

        # Step 4: Send sync sequence on all lanes (this still needs async for bit-level timing)
        # This step is now coordinated by _send_sync_sequence_all_lanes
        await self._send_sync_sequence_all_lanes()

    async def stop_packet_transmission(self):
        """Stop packet transmission (HS exit on all lanes) with parallel timing"""
        self.logger.info("TX PHY: Stopping packet transmission with parallel timing")

        # Step 1: Set HS-0 state on all lanes simultaneously for trail period
        for lane in self.data_lanes:
            self._hs_exit_sequence_step1_hs0(lane)

        # Wait for HS trail period
        await Timer(int(self.phy_config.t_hs_trail), units='ns')
        self.logger.info(f"TX PHY: HS trail duration complete ({self.phy_config.t_hs_trail}ns)")

        # Step 2: Set LP-11 state on all lanes simultaneously
        for lane in self.data_lanes:
            self._hs_exit_sequence_step2_lp11(lane)

        # Wait for HS exit period
        await Timer(int(self.phy_config.t_hs_exit), units='ns')
        self.logger.info(f"TX PHY: HS exit duration complete ({self.phy_config.t_hs_exit}ns)")

        # Handle clock lane if not continuous
        if not self.config.continuous_clock:
            await self._stop_hs_transmission(self.clock_lane)

    def get_transmission_statistics(self) -> dict:
        """Get transmission statistics"""
        return {
            'packet_count': self.packet_count,
            'bytes_transmitted': self.bytes_transmitted,
            'transmission_active': self.transmission_active,
            'lane_count': len(self.data_lanes),
            'continuous_clock': self.config.continuous_clock
        }

    def reset_transmission_statistics(self):
        """Reset transmission statistics"""
        self.packet_count = 0
        self.bytes_transmitted = 0
        self.transmission_active = False


class DPhyRxLaneState(Enum):
    LP_11 = 0
    LP_01 = 1
    LP_00 = 2
    HS_SYNC = 3
    HS_DATA = 4
    HS_TRAIL = 5


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

        # RX state machine per lane
        self.lane_rx_state = {i: DPhyRxLaneState.LP_11 for i in range(config.lane_count)}

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

        old_state = self.lane_rx_state[lane_idx]
        new_state = old_state

        # RX state machine - only transition if state actually changes
        if old_state == DPhyRxLaneState.LP_11:
            if p_val == 0 and n_val == 1:
                new_state = DPhyRxLaneState.LP_01
            elif p_val == 0 and n_val == 0:
                new_state = DPhyRxLaneState.LP_00
        elif old_state == DPhyRxLaneState.LP_01:
            if p_val == 0 and n_val == 0:
                new_state = DPhyRxLaneState.LP_00
            elif p_val == 1 and n_val == 1:
                new_state = DPhyRxLaneState.LP_11
        elif old_state == DPhyRxLaneState.LP_00:
            # In LP_00, we can transition to HS mode or back to LP_11
            if p_val == 1 and n_val == 1:
                new_state = DPhyRxLaneState.LP_11
            elif (p_val == 0 and n_val == 1) or (p_val == 1 and n_val == 0):
                # This is the start of HS mode - transition to HS_SYNC
                new_state = DPhyRxLaneState.HS_SYNC
        elif old_state == DPhyRxLaneState.HS_SYNC:
            # Stay in HS_SYNC during sync sequence, only transition to HS_DATA
            # after we've detected the sync pattern
            if self.lane_sync_aligned[lane_idx]:
                new_state = DPhyRxLaneState.HS_DATA
            # In multi-lane mode, also transition to HS_DATA if any lane has detected sync
            elif self.config.lane_distribution_enabled and len(self.data_lanes) > 1:
                # Check if any lane has detected sync
                any_lane_sync_aligned = any(self.lane_sync_aligned[i] for i in range(len(self.data_lanes)))
                if any_lane_sync_aligned:
                    new_state = DPhyRxLaneState.HS_DATA
        elif old_state == DPhyRxLaneState.HS_DATA:
            # In HS_DATA, we can transition back to LP_11 when HS mode ends
            if p_val == 1 and n_val == 1:
                new_state = DPhyRxLaneState.LP_11

        # Only handle state transition if state actually changed
        if new_state != old_state:
            await self._handle_state_transition(lane_idx, old_state, new_state)
            self.lane_rx_state[lane_idx] = new_state
            self.logger.info(f"Lane {lane_idx}: State ------------------------------------------------------ transition {old_state.name} -> {new_state.name}")

        # Log sampled values during HS_SYNC
        # if self.lane_rx_state[lane_idx] == DPhyRxLaneState.HS_SYNC:
            # self.logger.info(f"Lane {lane_idx}: HS_SYNC sample: data{lane_idx}_p={p_val}, data{lane_idx}_n={n_val}")

        # Only sample bits in HS_DATA state
        if self.lane_rx_state[lane_idx] == DPhyRxLaneState.HS_DATA:
            bit_value = (p_val == 1 and n_val == 0)
            await self._handle_data_bit(lane_idx, bit_value)
        elif self.lane_rx_state[lane_idx] == DPhyRxLaneState.HS_SYNC:
            # Also process bits in HS_SYNC state to detect sync sequence
            bit_value = (p_val == 1 and n_val == 0)
            await self._handle_data_bit(lane_idx, bit_value)

        # Only clear buffers when transitioning TO LP_11, not when already in LP_11
        if new_state == DPhyRxLaneState.LP_11 and old_state != DPhyRxLaneState.LP_11:
            self.logger.info(f"Lane {lane_idx}: Transitioned to LP_11, clearing RX buffers and state.")
            # Call packet end callback before clearing buffers
            if self.lane_processing_packet[lane_idx]:
                await self._handle_packet_end(lane_idx)

            self.lane_buffers[lane_idx].clear()
            self.lane_processing_packet[lane_idx] = False
            self.lane_sync_aligned[lane_idx] = False
            self.lane_packet_lengths[lane_idx] = 0
            self.lane_byte_buffers[lane_idx] = 0
            self.lane_bit_counts[lane_idx] = 0
            self.lane_bit_shifters[lane_idx] = 0

    async def _handle_state_transition(self, lane_idx: int, old_state: DPhyRxLaneState, new_state: DPhyRxLaneState):
        """Handle D-PHY state transitions"""
        self.logger.info(f"Lane {lane_idx}: State transition {old_state.name} -> {new_state.name}")
        # No need to set lane_lp_active or lane_hs_active; all state is in lane_rx_state now.
        # Add any additional logging or actions here if needed for future debugging.

    async def _handle_data_bit(self, lane_idx: int, bit_value: bool):
        """Handle received data bit from a specific lane."""
        if not self.lane_sync_aligned[lane_idx]:
            prev_shifter = self.lane_bit_shifters[lane_idx]
            self.lane_bit_shifters[lane_idx] = ((self.lane_bit_shifters[lane_idx] >> 1) | (bit_value << 15)) & 0xFFFF
            # self.logger.info(f"Lane {lane_idx}: HS_SYNC bit sampled: {int(bit_value)}, shifter before: 0x{prev_shifter:04x}, after: 0x{self.lane_bit_shifters[lane_idx]:04x}")
            found_sync = False
            for shift in range(9):
                shifted_bits = (self.lane_bit_shifters[lane_idx] >> (8 - shift)) & 0xFF
                # if (shifted_bits != 0): self.logger.info(f"Lane {lane_idx}: Checking for sync at shift {shift}: bits=0x{shifted_bits:02x} (pattern 0x{self.sync_sequence_bits:02x})")
                # exit(0)
                if shifted_bits == self.sync_sequence_bits:
                    found_sync = True
                    self.lane_sync_aligned[lane_idx] = True
                    self.lane_sync_shift[lane_idx] = shift
                    self.lane_processing_packet[lane_idx] = True
                    self.logger.info(f"Lane {lane_idx}: Sync detected at shift {shift} (pattern 0x{self.sync_sequence_bits:02x})")
                    # exit(0)
                    self.lane_byte_buffers[lane_idx] = 0
                    self.lane_bit_counts[lane_idx] = 0

                    # In multi-lane mode, all lanes should detect sync
                    # For now, we'll use the first lane to trigger packet start
                    is_first_lane_to_sync = not self.hs_active
                    if is_first_lane_to_sync:
                        self.hs_active = True
                        if self.on_packet_start:
                            await cocotb.start(self.on_packet_start())
                    else:
                        self.logger.debug(f"Lane {lane_idx}: Additional sync detected in multi-lane mode")
                    break
            # if not found_sync:
                # self.logger.info(f"Lane {lane_idx}: No sync detected in this bit sample. Current shifter: 0x{self.lane_bit_shifters[lane_idx]:04x}")
        else:
            self.lane_byte_buffers[lane_idx] = (self.lane_byte_buffers[lane_idx] >> 1) | (bit_value << 7)
            self.lane_bit_counts[lane_idx] += 1

            if self.lane_bit_counts[lane_idx] == 8:
                await self._process_byte(lane_idx, self.lane_byte_buffers[lane_idx])
                self.lane_byte_buffers[lane_idx] = 0
                self.lane_bit_counts[lane_idx] = 0

    async def _process_byte(self, lane_idx: int, byte_val: int):
        """Process a received byte from a specific lane."""
        if self.lane_processing_packet[lane_idx]:
            self.lane_buffers[lane_idx].append(byte_val)
            self.logger.debug(f"Lane {lane_idx}: Received byte 0x{byte_val:02x}")

            if self.config.lane_distribution_enabled and len(self.data_lanes) > 1:
                # Multi-lane mode: handle distributed data
                self.logger.info(f"Lane {lane_idx}: Calling _handle_distributed_data (multi-lane mode)")
                await self._handle_distributed_data()
            else:
                # Single-lane mode: handle complete packets on this lane
                await self._handle_single_lane_data(lane_idx, byte_val)

    async def _handle_single_lane_data(self, lane_idx: int, byte_val: int):
        """Handle data reception for single-lane mode"""
        # Only process data if we're in HS_DATA state and processing a packet
        if self.lane_rx_state[lane_idx] != DPhyRxLaneState.HS_DATA or not self.lane_processing_packet[lane_idx]:
            return

        # Check if this is the first byte and indicates a long packet
        if len(self.lane_buffers[lane_idx]) == 1:
            if _is_short_packet_type(byte_val):
                self.lane_packet_lengths[lane_idx] = 4
                self.logger.debug(f"Lane {lane_idx}: Short packet detected (4 bytes)")
            else:
                self.lane_packet_lengths[lane_idx] = -1
                self.logger.debug(f"Lane {lane_idx}: Long packet detected (length TBD)")

        # For long packets, determine length from header
        if self.lane_packet_lengths[lane_idx] == -1 and len(self.lane_buffers[lane_idx]) >= 4:
            try:
                header_bytes = bytes(self.lane_buffers[lane_idx][:4])
                header = Csi2PacketHeader.from_bytes(header_bytes)
                self.lane_packet_lengths[lane_idx] = 4 + header.word_count + 2  # Header + Data + CRC
                self.logger.debug(f"Lane {lane_idx}: Long packet length determined: {self.lane_packet_lengths[lane_idx]} bytes")
            except Exception as e:
                self.logger.warning(f"Lane {lane_idx}: Failed to parse header: {e}")
                self.lane_packet_lengths[lane_idx] = 4

        # Check if packet is complete
        if self.lane_packet_lengths[lane_idx] > 0 and len(self.lane_buffers[lane_idx]) >= self.lane_packet_lengths[lane_idx]:
            packet_data = bytes(self.lane_buffers[lane_idx][:self.lane_packet_lengths[lane_idx]])
            self.logger.info(f"Lane {lane_idx}: Packet complete ({len(packet_data)} bytes)")

            if self.on_data_received:
                self.logger.info(f"Lane {lane_idx}: Calling on_data_received callback with {len(packet_data)} bytes")
                await cocotb.start(self.on_data_received(packet_data))

            # Call packet end callback
            await self._handle_packet_end(lane_idx)

            # Reset for next packet and stop processing this lane until next sync
            self.lane_buffers[lane_idx] = self.lane_buffers[lane_idx][self.lane_packet_lengths[lane_idx]:]
            self.lane_packet_lengths[lane_idx] = 0
            self.lane_processing_packet[lane_idx] = False
            self.lane_sync_aligned[lane_idx] = False
            self.logger.debug(f"Lane {lane_idx}: Stopped processing until next sync")

    async def _handle_distributed_data(self):
        """Handle data reception for multi-lane distribution mode"""
        # For multi-lane mode, we need to reconstruct the original packet
        # from distributed data across all lanes

        # Get all lanes that are processing packets
        active_lanes = [i for i in range(len(self.data_lanes)) if self.lane_processing_packet[i]]
        if not active_lanes:
            self.logger.warning(f"Multi-lane: No active lanes found")
            return

        # For now, let's use a simpler approach: reconstruct from all enabled lanes
        # that have received data, regardless of sync alignment
        lanes_with_data = [i for i in range(len(self.data_lanes)) if len(self.lane_buffers[i]) > 0]

        if not lanes_with_data:
            self.logger.warning(f"Multi-lane: No lanes with data found")
            return

        # Debug: log what we have
        self.logger.info(f"Multi-lane: Active lanes: {active_lanes}, Lanes with data: {lanes_with_data}")

        # Log detailed state for all lanes
        for lane_idx in range(len(self.data_lanes)):
            self.logger.info(f"Multi-lane: Lane {lane_idx} - "
                           f"processing_packet: {self.lane_processing_packet[lane_idx]}, "
                           f"sync_aligned: {self.lane_sync_aligned[lane_idx]}, "
                           f"buffer_size: {len(self.lane_buffers[lane_idx])}, "
                           f"rx_state: {self.lane_rx_state[lane_idx].name}")

        for lane_idx in lanes_with_data:
            self.logger.info(f"Multi-lane: Lane {lane_idx} has {len(self.lane_buffers[lane_idx])} bytes: {[f'0x{b:02x}' for b in self.lane_buffers[lane_idx]]}")

        # Reconstruct the distributed data using the same algorithm as TX
        # TX distributes bytes round-robin: byte 0->lane 0, byte 1->lane 1, etc.
        reconstructed_data = []
        max_bytes = max(len(self.lane_buffers[i]) for i in lanes_with_data)

        self.logger.info(f"Multi-lane: Reconstructing with max_bytes={max_bytes}")

        for byte_idx in range(max_bytes):
            for lane_idx in range(len(self.data_lanes)):
                if lane_idx in lanes_with_data and byte_idx < len(self.lane_buffers[lane_idx]):
                    reconstructed_data.append(self.lane_buffers[lane_idx][byte_idx])

        self.logger.info(f"Multi-lane: Reconstructed {len(reconstructed_data)} bytes: {[f'0x{b:02x}' for b in reconstructed_data]}")

        if len(reconstructed_data) >= 1:
            first_byte = reconstructed_data[0]
            if _is_short_packet_type(first_byte):
                expected_length = 4
                self.logger.info(f"Multi-lane: Short packet detected (4 bytes total)")
            else:
                expected_length = -1
                self.logger.info(f"Multi-lane: Long packet detected (length TBD)")

            # For long packets, determine length from header once we have 4 bytes
            if expected_length == -1 and len(reconstructed_data) >= 4:
                try:
                    header_bytes = bytes(reconstructed_data[:4])
                    header = Csi2PacketHeader.from_bytes(header_bytes)
                    expected_length = 4 + header.word_count + 2  # Header + Data + CRC
                    self.logger.info(f"Multi-lane: Long packet length determined: {expected_length} bytes")
                except Exception as e:
                    self.logger.warning(f"Multi-lane: Failed to parse header: {e}")
                    expected_length = 4

            # Check if packet is complete
            self.logger.info(f"Multi-lane: Checking packet completion - reconstructed: {len(reconstructed_data)}, expected: {expected_length}")
            if expected_length > 0 and len(reconstructed_data) >= expected_length:
                packet_data = bytes(reconstructed_data[:expected_length])
                self.logger.info(f"Multi-lane: Packet complete ({len(packet_data)} bytes)")

                if self.on_data_received:
                    self.logger.info(f"Multi-lane: Calling on_data_received callback with {len(packet_data)} bytes")
                    await cocotb.start(self.on_data_received(packet_data))

                # Call packet end callback
                await self._handle_packet_end(0)  # Use lane 0 as representative

                # Remove processed bytes from all lane buffers
                # Calculate how many bytes each lane contributed
                total_bytes = len(reconstructed_data)
                bytes_per_lane = total_bytes // len(self.data_lanes)
                extra_bytes = total_bytes % len(self.data_lanes)

                for lane_idx in range(len(self.data_lanes)):
                    if lane_idx in lanes_with_data:
                        # Each lane contributed bytes_per_lane + 1 if it was one of the first extra_bytes lanes
                        bytes_to_remove = bytes_per_lane + (1 if lane_idx < extra_bytes else 0)
                        if bytes_to_remove > 0:
                            self.lane_buffers[lane_idx] = self.lane_buffers[lane_idx][bytes_to_remove:]
                            self.logger.debug(f"Multi-lane: Removed {bytes_to_remove} bytes from lane {lane_idx}")
            else:
                self.logger.warning(f"Multi-lane: Packet not complete - need {expected_length} bytes, have {len(reconstructed_data)}")
        else:
            self.logger.warning(f"Multi-lane: No reconstructed data available")

    async def _handle_packet_end(self, lane_idx: int):
        """Handle end of packet on a specific lane."""
        if self.lane_processing_packet[lane_idx]:
            self.logger.info(f"Lane {lane_idx}: Packet end detected")

            # Clear all buffers and reset state for this lane
            self.lane_buffers[lane_idx].clear()
            self.lane_processing_packet[lane_idx] = False
            self.lane_sync_aligned[lane_idx] = False
            self.lane_packet_lengths[lane_idx] = 0
            self.lane_byte_buffers[lane_idx] = 0
            self.lane_bit_counts[lane_idx] = 0
            self.lane_bit_shifters[lane_idx] = 0

            # Check if this is the last active lane
            active_lanes = [i for i in self.enabled_lanes if self.lane_processing_packet[i]]
            if not active_lanes:
                self.hs_active = False
                if self.on_packet_end:
                    self.logger.info(f"Lane {lane_idx}: Calling on_packet_end callback")
                    await cocotb.start(self.on_packet_end())
                else:
                    self.logger.warning(f"Lane {lane_idx}: on_packet_end callback not set")

    def reset_lane_state(self, lane_idx: int):
        """Reset state for a specific lane."""
        self.lane_buffers[lane_idx] = []
        self.lane_bit_shifters[lane_idx] = 0
        self.lane_byte_buffers[lane_idx] = 0
        self.lane_bit_counts[lane_idx] = 0
        self.lane_sync_aligned[lane_idx] = False
        self.lane_sync_shift[lane_idx] = 0
        self.lane_processing_packet[lane_idx] = False
        self.lane_packet_lengths[lane_idx] = 0

    def reset(self):
        """Reset all lane states."""
        for lane_idx in range(self.config.lane_count):
            self.reset_lane_state(lane_idx)
        self.hs_active = False

    def enable_lane(self, lane_idx: int):
        """Enable monitoring for a specific lane."""
        self.enabled_lanes.add(lane_idx)

    def disable_lane(self, lane_idx: int):
        """Disable monitoring for a specific lane."""
        self.enabled_lanes.discard(lane_idx)

    def get_received_data(self) -> bytes:
        """Get received data, merging from all lanes if lane distribution is enabled"""
        if self.config.lane_distribution_enabled and len(self.data_lanes) > 1:
            # Multi-lane: merge data from all lanes
            lane_data = []
            for i in range(len(self.data_lanes)):
                if i in self.lane_buffers:
                    lane_data.append(bytes(self.lane_buffers[i]))
                else:
                    lane_data.append(b'')

            # Merge lanes back to original byte stream
            from ..utils import lanes_to_bytes
            return lanes_to_bytes(lane_data)
        else:
            # Single-lane operation
            if 0 in self.lane_buffers:
                return bytes(self.lane_buffers[0])
            return b''

    def set_rx_callbacks(self, on_packet_start=None, on_packet_end=None, on_data_received=None):
        """Set receiver callbacks."""
        self.on_packet_start = on_packet_start
        self.on_packet_end = on_packet_end
        self.on_data_received = on_data_received