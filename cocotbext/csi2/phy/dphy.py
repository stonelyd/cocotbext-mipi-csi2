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
from cocotb.triggers import Timer, RisingEdge, FallingEdge, Edge, First
from cocotb.clock import Clock
import asyncio
from typing import List, Optional, Callable
import logging
from enum import Enum
from ..config import Csi2Config, Csi2PhyConfig
from ..bus import Csi2DPhyBus
from ..exceptions import Csi2PhyError, Csi2TimingError


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

    async def _hs_prepare_sequence(self):
        """Execute HS prepare sequence with improved timing"""
        self.logger.info(f"Lane {self.lane.name}: Starting HS prepare sequence")

        #set LP-01 state for 50ns
        self._set_lp_state(DPhyState.LP_01)
        await Timer(50, units='ns')
        self.logger.info(f"Lane {self.lane.name}: Set LP-01 state for {self.phy_config.t_lpx}ns")

        # Set LP-00 state
        self._set_lp_state(DPhyState.LP_00)
        self.logger.info(f"Lane {self.lane.name}: Set LP-00 state for {self.phy_config.t_hs_prepare}ns")

        # Wait for HS prepare time
        await Timer(int(self.phy_config.t_hs_prepare), units='ns')

        # Set HS-0 state for HS zero period
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0
        self.logger.info(f"Lane {self.lane.name}: Set HS-0 state for {self.phy_config.t_hs_zero}ns")

        # Wait for HS zero period
        await Timer(int(self.phy_config.t_hs_zero), units='ns')

        # Insert HS Sync-Sequence '00011101' (0x1D) according to D-PHY v2.5 Section 6.4.2
        await self._send_sync_sequence()

        self.hs_active = True
        self.lp_active = False
        self.logger.info(f"Lane {self.lane.name}: HS prepare sequence complete, ready for data")

    async def _hs_exit_sequence(self):
        """Execute HS exit sequence"""
        self.logger.debug(f"Lane {self.lane.name}: Starting HS exit")

        # HS trail
        await Timer(self.phy_config.t_hs_trail, units='ns')

        # LP-11 (HS exit)
        self._set_lp_state(DPhyState.LP_11)
        self.hs_active = False
        self.lp_active = True
        await Timer(self.phy_config.t_hs_exit, units='ns')

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

    async def send_lp_sequence(self, sequence: List[int]):
        """Send LP sequence"""
        for state in sequence:
            self._set_lp_state(state)
            await Timer(100, units='ns')  # LP state duration

    async def start_hs_transmission(self):
        """Start HS transmission on this lane"""
        await self._hs_prepare_sequence()

    async def stop_hs_transmission(self):
        """Stop HS transmission on this lane"""
        await self._hs_exit_sequence()


class DPhyRxReceiver:
    """D-PHY Receiver Model"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, lane: DPhyLane):
        """
        Initialize D-PHY receiver

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
        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.rx.{lane.name}')
        self.logger.info(f"Lane {self.lane.name}: RX signal handles - p: {self.sig_p.value}, n: {self.sig_n.value}")

        # Log initial signal state
        try:
            initial_p = int(self.sig_p.value)
            initial_n = int(self.sig_n.value)
            self.logger.info(f"Lane {self.lane.name}: RX initial signal state - p: {initial_p}, n: {initial_n}")
        except (ValueError, TypeError):
            self.logger.info(f"Lane {self.lane.name}: RX initial signal state - p: z, n: z")

        # Callbacks
        self.on_hs_start = None
        self.on_hs_end = None
        self.on_data_received = None

        # Timing tracking for automatic LP_00 to HS transition
        self.lp00_entry_time = None
        self.hs_transition_task = None
        self.ui_ns = 1000.0 / config.bit_rate_mbps  # Unit interval in nanoseconds

    def _decode_lp_state(self) -> int:
        """Decode current LP state from signals"""
        try:
            p_val = int(self.sig_p.value)
            n_val = int(self.sig_n.value)
        except ValueError:
            # Handle 'z' (high impedance) values - treat as LP-11
            return DPhyState.LP_11

        # Debug: log signal values occasionally (reduced frequency)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 1000 == 0:  # Log every 1000th call instead of 100th
            self.logger.debug(f"Lane {self.lane.name}: Signal values p={p_val}, n={n_val}")

        if p_val == 0 and n_val == 0:
            return DPhyState.LP_00
        elif p_val == 0 and n_val == 1:
            return DPhyState.LP_01
        elif p_val == 1 and n_val == 0:
            return DPhyState.LP_10
        elif p_val == 1 and n_val == 1:
            return DPhyState.LP_11
        else:
            # Invalid state - treat as LP-11
            return DPhyState.LP_11

    def _decode_hs_state(self) -> int:
        """Decode current HS state from signals"""
        try:
            p_val = int(self.sig_p.value)
            n_val = int(self.sig_n.value)
        except ValueError:
            # Handle 'z' (high impedance) values - treat as invalid
            return -1

        # HS-0: p=0, n=1; HS-1: p=1, n=0
        if p_val == 0 and n_val == 1:
            return DPhyState.HS_0
        elif p_val == 1 and n_val == 0:
            return DPhyState.HS_1
        else:
            # Invalid HS state
            return -1

    def _decode_current_state(self) -> int:
        """Decode current state based on current mode (LP or HS)"""
        if self.hs_active:
            # In HS mode, decode HS state
            return self._decode_hs_state()
        else:
            # In LP mode, decode LP state
            return self._decode_lp_state()

    def _is_valid_state_transition(self, from_state: int, to_state: int) -> bool:
        """Validate state transition according to D-PHY protocol rules"""
        # LP_00 can only transition to HS states (HS_0 or HS_1)
        if from_state == DPhyState.LP_00:
            return to_state in [DPhyState.HS_0, DPhyState.HS_1]

        # HS states can transition to any LP state (typically LP_11 for exit)
        if from_state in [DPhyState.HS_0, DPhyState.HS_1]:
            return to_state in [DPhyState.LP_00, DPhyState.LP_01, DPhyState.LP_10, DPhyState.LP_11]

        # Other LP states can transition to LP_00 (for HS entry preparation) or other LP states
        if from_state in [DPhyState.LP_01, DPhyState.LP_10, DPhyState.LP_11]:
            return to_state in [DPhyState.LP_00, DPhyState.LP_01, DPhyState.LP_10, DPhyState.LP_11]

        # Invalid from_state
        return False

    async def _schedule_hs_transition(self):
        """Schedule automatic transition from LP_00 to HS after timing delay"""
        # Calculate delay: 86ns + 6*UI
        delay_ns = 86 + 6 * self.ui_ns
        self.logger.info(f"Lane {self.lane.name}: Scheduling HS transition in {delay_ns:.2f}ns (86ns + 6*{self.ui_ns:.2f}ns)")

        # Wait for the calculated delay
        await Timer(int(delay_ns), units='ns')

        # Check if we're still in LP_00 state (transition hasn't been cancelled)
        if self.current_state == DPhyState.LP_00 and not self.hs_active:
            self.logger.info(f"Lane {self.lane.name}: Executing automatic LP_00 -> HS transition")

            # Set HS mode
            self.hs_active = True
            self.lp_active = False

            # Trigger HS start callback
            if self.on_hs_start:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self.on_hs_start):
                    await self.on_hs_start()
                else:
                    self.on_hs_start()
        else:
            self.logger.debug(f"Lane {self.lane.name}: HS transition cancelled (no longer in LP_00 or already in HS mode)")

    async def update_state(self):
        """Update current state and detect HS mode transitions"""
        previous_state = self.current_state
        previous_hs_active = self.hs_active

        # Decode current state
        new_state = self._decode_current_state()

        # Validate state transition before updating
        if new_state != -1:  # Valid state detected
            if self._is_valid_state_transition(previous_state, new_state):
                self.current_state = new_state

                # Handle LP_00 entry - schedule automatic HS transition
                if new_state == DPhyState.LP_00 and previous_state != DPhyState.LP_00:
                    self.logger.info(f"Lane {self.lane.name}: Entered LP_00, scheduling automatic HS transition")

                    # Cancel any existing transition task
                    if self.hs_transition_task and not self.hs_transition_task.done():
                        self.hs_transition_task.cancel()

                    # Schedule new HS transition
                    self.hs_transition_task = cocotb.start_soon(self._schedule_hs_transition())

                # Detect immediate HS mode transitions (from signal changes)
                elif not self.hs_active and new_state in [DPhyState.HS_0, DPhyState.HS_1]:
                    # Cancel any pending automatic transition
                    if self.hs_transition_task and not self.hs_transition_task.done():
                        self.hs_transition_task.cancel()

                    # Entering HS mode immediately
                    self.hs_active = True
                    self.lp_active = False
                    self.logger.info(f"Lane {self.lane.name}: Entering HS mode immediately (state: {DPhyState.name(new_state)})")

                    if self.on_hs_start:
                        # Handle both sync and async callbacks
                        if asyncio.iscoroutinefunction(self.on_hs_start):
                            await self.on_hs_start()
                        else:
                            self.on_hs_start()

                elif self.hs_active and new_state in [DPhyState.LP_00, DPhyState.LP_01, DPhyState.LP_10, DPhyState.LP_11]:
                    # Exiting HS mode
                    self.hs_active = False
                    self.lp_active = True
                    self.logger.info(f"Lane {self.lane.name}: Exiting HS mode, entering LP mode (state: {DPhyState.name(new_state)})")

                    if self.on_hs_end:
                        # Handle both sync and async callbacks
                        if asyncio.iscoroutinefunction(self.on_hs_end):
                            await self.on_hs_end()
                        else:
                            self.on_hs_end()

                # Log state changes (but suppress HS-0/HS-1 transitions to reduce noise)
                if previous_state != new_state and not (self.hs_active and new_state in [DPhyState.HS_0, DPhyState.HS_1]):
                    self.logger.info(f"Lane {self.lane.name}: State change {DPhyState.name(previous_state)} -> {DPhyState.name(new_state)}")
            # else:
            #     # Invalid state transition - log warning but don't update state
            #     self.logger.warning(f"Lane {self.lane.name}: Invalid state transition {DPhyState.name(previous_state)} -> {DPhyState.name(new_state)} (ignored)")


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
        """Start packet transmission (HS prepare on all lanes)"""
        self.logger.info("TX PHY: Starting packet transmission")

        if not self.config.continuous_clock:
            await self.clock_tx.start_hs_transmission()

        # Start all data lanes sequentially (cocotb compatibility)
        for tx_lane in self.data_tx:
            await tx_lane.start_hs_transmission()

    async def stop_packet_transmission(self):
        """Stop packet transmission (HS exit on all lanes)"""
        self.logger.info("TX PHY: Stopping packet transmission")

        # Stop all data lanes sequentially (cocotb compatibility)
        for tx_lane in self.data_tx:
            await tx_lane.stop_hs_transmission()

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

        # Create lanes with clear identification
        self.clock_lane = DPhyLane(DPhyLaneType.CLOCK)  # index ignored
        self.data_lanes = [DPhyLane(DPhyLaneType.DATA, i) for i in range(config.lane_count)]

        # Create receivers
        self.clock_rx = DPhyRxReceiver(bus, config, self.phy_config, self.clock_lane)
        self.data_rx = [DPhyRxReceiver(bus, config, self.phy_config, lane) for lane in self.data_lanes]

        # For backward compatibility, create rx_lanes list (only data lanes)
        self.rx_lanes = self.data_rx

        self.logger = logging.getLogger('cocotbext.csi2.dphy.rx_model')

        # Data monitoring state
        self.hs_active = False
        self.packet_active = False

        # Track HS state for each lane
        self.lane_hs_active = {i: False for i in range(config.lane_count)}

        # Data buffers for each lane
        self.lane_buffers = {i: bytearray() for i in range(config.lane_count)}
        self.lane_byte_buffers = {i: 0 for i in range(config.lane_count)}
        self.lane_bit_counts = {i: 0 for i in range(config.lane_count)}

        # Protocol overhead tracking
        self.sync_sequence_detected = {i: False for i in range(config.lane_count)}
        self.overhead_bytes_skipped = {i: 0 for i in range(config.lane_count)}
        self.sync_sequence = 0xB8  # D-PHY sync sequence (actual received value)

        # Lane enable status
        self.enabled_lanes = set(range(config.lane_count))  # All lanes enabled by default

        # Callbacks for the overall model
        self.on_hs_start = None
        self.on_hs_end = None
        self.on_data_received = None

        # Start clock monitoring
        self._clock_monitor_task = cocotb.start_soon(self._monitor_clock_events())

    async def _check_all_lanes_hs_active(self):
        """Check if all enabled lanes are in HS state and update overall hs_active"""
        all_hs_active = all(self.lane_hs_active[lane_idx] for lane_idx in self.enabled_lanes)

        if all_hs_active and not self.hs_active:
            self.hs_active = True
            self.logger.info("All enabled data lanes in HS state - setting overall HS active")

            # Trigger HS start callback
            if self.on_hs_start:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self.on_hs_start):
                    await self.on_hs_start()
                else:
                    self.on_hs_start()

        elif not all_hs_active and self.hs_active:
            self.hs_active = False
            self.logger.info("Not all enabled data lanes in HS state - setting overall HS inactive")

            # Trigger HS end callback
            if self.on_hs_end:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self.on_hs_end):
                    await self.on_hs_end()
                else:
                    self.on_hs_end()


    async def _monitor_clock_events(self):
        """Monitor clock lane for positive/negative edge events"""
        self.logger.info("Starting clock event monitoring")

        # Track previous signal values for edge detection
        prev_p = int(self.clock_rx.sig_p.value) if hasattr(self.clock_rx.sig_p, 'value') else 0
        prev_n = int(self.clock_rx.sig_n.value) if hasattr(self.clock_rx.sig_n, 'value') else 0

        while True:
            try:
                # Wait for any edge on either signal
                await First(
                    RisingEdge(self.clock_rx.sig_p),
                    FallingEdge(self.clock_rx.sig_p),
                    RisingEdge(self.clock_rx.sig_n),
                    FallingEdge(self.clock_rx.sig_n)
                )

                # Get current signal values
                current_p = int(self.clock_rx.sig_p.value)
                current_n = int(self.clock_rx.sig_n.value)

                # Only process if signals actually changed
                if current_p != prev_p or current_n != prev_n:
                    current_time = cocotb.utils.get_sim_time('ns')

                    # Detect clock events: p going positive, n going negative
                    if prev_p == 0 and current_p == 1:  # p going positive
                        self.logger.debug(f"Clock event: p going positive at {current_time}ns")

                        # Sample all enabled data lanes on clock event
                        for lane_idx in self.enabled_lanes:
                            await self._sample_data_lane(lane_idx)

                    if prev_n == 1 and current_n == 0:  # n going negative
                        self.logger.debug(f"Clock event: n going negative at {current_time}ns")

                        # Sample all enabled data lanes on clock event
                        for lane_idx in self.enabled_lanes:
                            await self._sample_data_lane(lane_idx)

                    # Update previous values
                    prev_p = current_p
                    prev_n = current_n

            except Exception as e:
                self.logger.error(f"Error in clock event monitoring: {e}")
                # Brief pause before retrying
                await Timer(1, units='ns')


    async def _sample_data_lane(self, lane_idx: int):
        """Sample a specific data lane for HS data"""

        await self.data_rx[lane_idx].update_state()

        # Check if this lane has transitioned to HS state
        lane_hs_active = self.data_rx[lane_idx].hs_active
        if lane_hs_active != self.lane_hs_active[lane_idx]:
            self.lane_hs_active[lane_idx] = lane_hs_active
            self.logger.info(f"Lane {lane_idx} HS state changed to: {lane_hs_active}")

            # Check if all enabled lanes are now in HS state
            await self._check_all_lanes_hs_active()

        # Only sample if data monitoring is active
        if not self.hs_active:
            return

        rx_lane = self.data_rx[lane_idx]

        try:
            p_val = int(rx_lane.sig_p.value)
            n_val = int(rx_lane.sig_n.value)
        except (ValueError, TypeError):
            # Handle invalid values
            return

        # HS-0: p=0, n=1; HS-1: p=1, n=0
        if p_val == 0 and n_val == 1:
            await self._handle_data_bit(lane_idx, False)
        elif p_val == 1 and n_val == 0:
            await self._handle_data_bit(lane_idx, True)
        elif p_val == 1 and n_val == 1:  # LP-11, end of packet
            await self._handle_packet_end(lane_idx)

    async def _handle_data_bit(self, lane_idx: int, bit_value: bool):
        """Handle received data bit from a specific lane"""
        # Accumulate bits into bytes (LSB first)
        if bit_value:
            self.lane_byte_buffers[lane_idx] |= (1 << self.lane_bit_counts[lane_idx])

        self.lane_bit_counts[lane_idx] += 1

        if self.lane_bit_counts[lane_idx] >= 8:
            # Complete byte received
            byte_val = self.lane_byte_buffers[lane_idx]

            # Check if this is the sync sequence
            if byte_val == self.sync_sequence and not self.sync_sequence_detected[lane_idx]:
                self.sync_sequence_detected[lane_idx] = True
                self.overhead_bytes_skipped[lane_idx] += 1
                self.logger.info(f"Lane {lane_idx}: Detected sync sequence 0x{byte_val:02x}, skipping protocol overhead")

                # Reset for next byte without forwarding
                self.lane_byte_buffers[lane_idx] = 0
                self.lane_bit_counts[lane_idx] = 0
                return

            # Skip a few bytes after sync sequence (protocol overhead)
            if self.sync_sequence_detected[lane_idx] and self.overhead_bytes_skipped[lane_idx] < 4:
                self.overhead_bytes_skipped[lane_idx] += 1
                self.logger.debug(f"Lane {lane_idx}: Skipping protocol overhead byte 0x{byte_val:02x} ({self.overhead_bytes_skipped[lane_idx]}/4)")

                # Reset for next byte without forwarding
                self.lane_byte_buffers[lane_idx] = 0
                self.lane_bit_counts[lane_idx] = 0
                return

            # After sync sequence and overhead, forward actual packet data
            self.lane_buffers[lane_idx].append(byte_val)
            self.logger.debug(f"Lane {lane_idx}: Received packet byte 0x{byte_val:02x}")

            # Forward data to RX callbacks
            if self.on_data_received:
                self.logger.info(f"Lane {lane_idx}: Forwarding packet byte 0x{byte_val:02x} to RX callback")
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self.on_data_received):
                    await self.on_data_received(byte_val)
                else:
                    self.on_data_received(byte_val)
            else:
                self.logger.warning(f"Lane {lane_idx}: No on_data_received callback set")

            # Reset for next byte
            self.lane_byte_buffers[lane_idx] = 0
            self.lane_bit_counts[lane_idx] = 0

    async def _handle_packet_end(self, lane_idx: int):
        """Handle end of packet on a specific lane"""
        if self.lane_buffers[lane_idx]:
            self.logger.info(f"Lane {lane_idx}: Packet end, {len(self.lane_buffers[lane_idx])} bytes received")

            # Process packet data for lane demuxing and decoding
            packet_data = bytes(self.lane_buffers[lane_idx])
            await self._process_packet_data(lane_idx, packet_data)

            # Forward remaining data to RX callbacks
            if self.on_data_received and packet_data:
                self.logger.info(f"Lane {lane_idx}: Forwarding {len(packet_data)} bytes to RX callback at packet end")
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(self.on_data_received):
                    await self.on_data_received(packet_data)
                else:
                    self.on_data_received(packet_data)
            elif not self.on_data_received:
                self.logger.warning(f"Lane {lane_idx}: No on_data_received callback set for packet end")

            # Clear buffer and reset sync detection for next packet
            self.lane_buffers[lane_idx].clear()
            self.sync_sequence_detected[lane_idx] = False
            self.overhead_bytes_skipped[lane_idx] = 0

    async def _process_packet_data(self, lane_idx: int, packet_data: bytes):
        """Process packet data for lane demuxing and packet decoding"""
        self.logger.info(f"Processing packet data from lane {lane_idx}: {len(packet_data)} bytes")

        # Here you would implement lane demuxing and packet decoding logic
        # For now, just log the data
        self.logger.info(f"Lane {lane_idx} packet data: {[f'0x{b:02x}' for b in packet_data]}")


    def enable_lane(self, lane_idx: int):
        """Enable a specific data lane"""
        if 0 <= lane_idx < self.config.lane_count:
            self.enabled_lanes.add(lane_idx)
            self.logger.info(f"Enabled lane {lane_idx}")
        else:
            self.logger.error(f"Invalid lane index: {lane_idx}")

    def disable_lane(self, lane_idx: int):
        """Disable a specific data lane"""
        if lane_idx in self.enabled_lanes:
            self.enabled_lanes.remove(lane_idx)
            self.logger.info(f"Disabled lane {lane_idx}")
        else:
            self.logger.error(f"Lane {lane_idx} is not enabled")

    def get_lane_buffer(self, lane_idx: int) -> bytes:
        """Get the current buffer contents for a specific lane"""
        if 0 <= lane_idx < self.config.lane_count:
            return bytes(self.lane_buffers[lane_idx])
        else:
            self.logger.error(f"Invalid lane index: {lane_idx}")
            return b''

    def clear_lane_buffer(self, lane_idx: int):
        """Clear the buffer for a specific lane"""
        if 0 <= lane_idx < self.config.lane_count:
            self.lane_buffers[lane_idx].clear()
            self.lane_byte_buffers[lane_idx] = 0
            self.lane_bit_counts[lane_idx] = 0
            self.logger.info(f"Cleared buffer for lane {lane_idx}")
        else:
            self.logger.error(f"Invalid lane index: {lane_idx}")

    def get_enabled_lanes(self) -> set:
        """Get the set of enabled lane indices"""
        return self.enabled_lanes.copy()

    def get_received_data(self) -> bytes:
        """Collect received data from all RX lanes"""
        if not self.data_rx:
            return b''

        if self.config.lane_distribution_enabled and len(self.data_rx) > 1:
            # Collect data from all lanes
            lane_data = []
            for rx_lane in self.data_rx:
                lane_data.append(rx_lane.get_received_data())

            # Merge lane data back into single stream
            from ..utils import lanes_to_bytes
            return lanes_to_bytes(lane_data)
        else:
            # Get data from the first lane only (no distribution)
            return self.data_rx[0].get_received_data()

    def set_rx_callbacks(self, on_packet_start=None, on_packet_end=None,
                        on_data_received=None):
        """Set callbacks for RX events"""
        # Set callbacks for the overall model
        self.on_hs_start = on_packet_start
        self.on_hs_end = on_packet_end
        self.on_data_received = on_data_received

        # Set callbacks for clock lane
        self.clock_rx.on_hs_start = on_packet_start
        self.clock_rx.on_hs_end = on_packet_end
        self.clock_rx.on_data_received = on_data_received

        # Set callbacks for all RX lanes
        for rx_lane in self.data_rx:
            rx_lane.on_hs_start = on_packet_start
            rx_lane.on_hs_end = on_packet_end
            rx_lane.on_data_received = on_data_received

    def get_signal_statistics(self) -> dict:
        """Get signal quality statistics from all lanes"""
        stats = {
            'clock_lane': {},  # No longer using signal monitor
            'data_lanes': []
        }

        for rx_lane in self.data_rx:
            stats['data_lanes'].append({})  # No longer using signal monitor

        return stats

    def reset_signal_monitors(self):
        """Reset signal quality monitoring on all lanes"""
        # No longer using signal monitors
        pass

    def reset_lane_hs_tracking(self):
        """Reset HS state tracking for all lanes"""
        for lane_idx in range(self.config.lane_count):
            self.lane_hs_active[lane_idx] = False
        self.hs_active = False
        self.logger.info("Reset lane HS state tracking")

    def reset_sync_sequence_detection(self):
        """Reset sync sequence detection for all lanes"""
        for lane_idx in range(self.config.lane_count):
            self.sync_sequence_detected[lane_idx] = False
            self.overhead_bytes_skipped[lane_idx] = 0
        self.logger.info("Reset sync sequence detection for all lanes")

    def get_lane_hs_status(self) -> dict:
        """Get HS status for all lanes"""
        return {
            'overall_hs_active': self.hs_active,
            'lane_hs_active': self.lane_hs_active.copy(),
            'enabled_lanes': list(self.enabled_lanes)
        }


# Legacy aliases for backward compatibility
DPhyTx = DPhyTxTransmitter
DPhyRx = DPhyRxReceiver
DPhyModel = DPhyTxModel  # Default to TX model for backward compatibility
