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
from ..config import Csi2Config, Csi2PhyConfig
from ..bus import Csi2DPhyBus
from ..exceptions import Csi2PhyError, Csi2TimingError


class DPhyState:
    """D-PHY Lane States"""
    LP_00 = 0  # Low Power 00
    LP_01 = 1  # Low Power 01
    LP_10 = 2  # Low Power 10
    LP_11 = 3  # Low Power 11
    HS_0 = 4   # High Speed 0
    HS_1 = 5   # High Speed 1


class DPhyTx:
    """D-PHY Transmitter Model"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, lane_index: int = 0):
        """
        Initialize D-PHY transmitter

        Args:
            bus: D-PHY bus interface
            config: CSI-2 configuration
            phy_config: PHY-specific configuration
            lane_index: Lane index (0 for clock, 1+ for data)
        """
        self.bus = bus
        self.config = config
        self.phy_config = phy_config
        self.lane_index = lane_index
        self.is_clock_lane = (lane_index == 0)

        # Internal state
        self.current_state = DPhyState.LP_11
        self.hs_active = False
        self.lp_active = True

        # Signal handles
        if self.is_clock_lane:
            self.sig_p = getattr(bus, 'clk_p', None)
            self.sig_n = getattr(bus, 'clk_n', None)
        else:
            self.sig_p = getattr(bus, f'data{lane_index-1}_p', None)
            self.sig_n = getattr(bus, f'data{lane_index-1}_n', None)

        if not self.sig_p or not self.sig_n:
            raise Csi2PhyError(f"Missing D-PHY signals for lane {lane_index}")

        # Timing parameters
        self.bit_period_ns = config.get_bit_period_ns()
        self.byte_period_ns = config.get_byte_period_ns()

        # Initialize signals
        self._set_lp_state(DPhyState.LP_11)

        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.tx.lane{lane_index}')

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

    async def _hs_prepare_sequence(self):
        """Execute HS prepare sequence"""
        self.logger.info(f"Lane {self.lane_index}: Starting HS prepare sequence")

        # LP-00 (HS prepare)
        self._set_lp_state(DPhyState.LP_00)
        self.logger.info(f"Lane {self.lane_index}: Set LP-00 state for {self.phy_config.t_hs_prepare}ns")
        await Timer(self.phy_config.t_hs_prepare, units='ns')

        # HS-0 (HS zero) - ensure proper differential signaling
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0
        self.hs_active = True
        self.lp_active = False
        self.logger.info(f"Lane {self.lane_index}: Set HS-0 state for {self.phy_config.t_hs_zero}ns")
        await Timer(self.phy_config.t_hs_zero, units='ns')

        self.logger.info(f"Lane {self.lane_index}: HS prepare sequence complete, ready for data")

    async def _hs_exit_sequence(self):
        """Execute HS exit sequence"""
        self.logger.debug(f"Lane {self.lane_index}: Starting HS exit")

        # HS trail
        await Timer(self.phy_config.t_hs_trail, units='ns')

        # LP-11 (HS exit)
        self._set_lp_state(DPhyState.LP_11)
        self.hs_active = False
        self.lp_active = True
        await Timer(self.phy_config.t_hs_exit, units='ns')

    async def send_hs_data(self, data: bytes):
        """
        Send data in High Speed mode

        Args:
            data: Byte data to transmit
        """
        if not self.hs_active:
            await self._hs_prepare_sequence()

        self.logger.info(f"Lane {self.lane_index}: Sending {len(data)} bytes in HS mode")
        self.logger.info(f"Lane {self.lane_index}: Data: {[f'{b:02x}' for b in data[:10]]}...")

        for byte_val in data:
            await self._send_hs_byte(byte_val)

        self.logger.info(f"Lane {self.lane_index}: HS data transmission complete")

    async def _send_hs_byte(self, byte_val: int):
        """Send single byte in HS mode"""
        self.logger.debug(f"Lane {self.lane_index}: Sending byte 0x{byte_val:02x}")

        for bit_pos in range(8):
            bit_val = (byte_val >> bit_pos) & 1

            if bit_val:
                self.sig_p.value = 1
                self.sig_n.value = 0
                self.current_state = DPhyState.HS_1
                self.logger.debug(f"Lane {self.lane_index}: Bit {bit_pos}: HS-1")
            else:
                self.sig_p.value = 0
                self.sig_n.value = 1
                self.current_state = DPhyState.HS_0
                self.logger.debug(f"Lane {self.lane_index}: Bit {bit_pos}: HS-0")

            await Timer(self.bit_period_ns, units='ns')

    async def send_lp_sequence(self, sequence: List[int]):
        """
        Send Low Power control sequence

        Args:
            sequence: List of LP states to send
        """
        if self.hs_active:
            await self._hs_exit_sequence()

        for state in sequence:
            self._set_lp_state(state)
            await Timer(self.phy_config.t_lpx, units='ns')

    async def start_hs_transmission(self):
        """Start High Speed transmission mode"""
        if not self.hs_active:
            await self._hs_prepare_sequence()

    async def stop_hs_transmission(self):
        """Stop High Speed transmission mode"""
        if self.hs_active:
            await self._hs_exit_sequence()


class DPhyRx:
    """D-PHY Receiver Model"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, lane_index: int = 0):
        """
        Initialize D-PHY receiver

        Args:
            bus: D-PHY bus interface
            config: CSI-2 configuration
            phy_config: PHY-specific configuration
            lane_index: Lane index (0 for clock, 1+ for data)
        """
        self.bus = bus
        self.config = config
        self.phy_config = phy_config
        self.lane_index = lane_index
        self.is_clock_lane = (lane_index == 0)

        # Internal state
        self.current_state = DPhyState.LP_11
        self.hs_active = False
        self.lp_active = True

        # Signal handles
        if self.is_clock_lane:
            self.sig_p = getattr(bus, 'clk_p', None)
            self.sig_n = getattr(bus, 'clk_n', None)
        else:
            self.sig_p = getattr(bus, f'data{lane_index-1}_p', None)
            self.sig_n = getattr(bus, f'data{lane_index-1}_n', None)

        if not self.sig_p or not self.sig_n:
            raise Csi2PhyError(f"Missing D-PHY signals for lane {lane_index}")

        # Data buffering
        self.received_data = bytearray()
        self.byte_buffer = 0
        self.bit_count = 0
        self.chunk_size = 16  # Send data in 16-byte chunks

        # Callbacks
        self.on_hs_start = None
        self.on_hs_end = None
        self.on_data_received = None

        # Monitoring tasks
        self._monitor_task = None
        self._hs_monitor_task = None

        # Signal quality monitoring
        self.signal_monitor = DPhySignalMonitor(lane_index)

        # Start signal monitoring only for data lanes (not clock lane)
        if not self.is_clock_lane:
            self._monitor_task = cocotb.start_soon(self._monitor_signals())

        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.rx.lane{lane_index}')

    def _decode_lp_state(self) -> int:
        """Decode current LP state from signals"""
        try:
            p_val = int(self.sig_p.value)
            n_val = int(self.sig_n.value)
        except ValueError:
            # Handle 'z' (high impedance) values - treat as LP-11
            return DPhyState.LP_11

        # Debug: log signal values occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter % 100 == 0:  # Log every 100th call
            self.logger.info(f"Lane {self.lane_index}: Signal values p={p_val}, n={n_val}")

        if p_val == 0 and n_val == 0:
            return DPhyState.LP_00
        elif p_val == 0 and n_val == 1:
            return DPhyState.LP_01
        elif p_val == 1 and n_val == 0:
            return DPhyState.LP_10
        else:  # p_val == 1 and n_val == 1
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

    async def _monitor_signals(self):
        """Monitor D-PHY signals for state changes with debouncing"""
        self.current_state = self._decode_lp_state()
        self.logger.info(f"Lane {self.lane_index}: Starting signal monitoring, initial state: {self.current_state}")

        # Debouncing parameters
        debounce_time = 10  # ns
        last_state_change = 0
        stable_state = self.current_state

        while True:
            try:
                # Use polling with debouncing instead of edge triggers
                await Timer(5, units='ns')  # Poll every 5ns

                current_time = cocotb.utils.get_sim_time('ns')
                new_state = self._decode_lp_state()

                # Only process state changes if enough time has passed (debouncing)
                if new_state != stable_state and (current_time - last_state_change) > debounce_time:
                    # Verify state is stable by checking again after a short delay
                    await Timer(2, units='ns')
                    verified_state = self._decode_lp_state()

                    if verified_state == new_state:
                        # State change is confirmed stable
                        self.signal_monitor.record_signal_transition(stable_state, new_state, current_time)
                        await self._process_state_change(stable_state, new_state)
                        stable_state = new_state
                        last_state_change = current_time
                        self.current_state = new_state
                    else:
                        # State change was transient, ignore
                        self.logger.debug(f"Lane {self.lane_index}: Ignoring transient state change {stable_state} -> {new_state}")

            except Exception as e:
                self.logger.error(f"Error in signal monitoring: {e}")
                await Timer(10, units='ns')

    async def _process_state_change(self, old_state: int, new_state: int):
        """Process D-PHY state transitions"""
        self.logger.info(f"Lane {self.lane_index}: State {old_state} -> {new_state}")

        # Detect HS prepare (any LP state -> LP-00)
        if self.lp_active and new_state == DPhyState.LP_00:
            self.logger.info(f"Lane {self.lane_index}: HS prepare detected!")
            await self._handle_hs_prepare()

        # Detect HS exit (back to LP-11)
        elif self.hs_active and new_state == DPhyState.LP_11:
            self.logger.info(f"Lane {self.lane_index}: HS exit detected!")
            await self._handle_hs_exit()

    async def _monitor_hs_data(self):
        """Monitor HS data when in HS mode with improved stability"""
        bit_period = self.config.get_bit_period_ns()
        sample_delay = bit_period * 0.4  # Sample at 40% of bit period for better timing

        self.logger.info(f"Lane {self.lane_index}: Starting HS data monitoring with {bit_period}ns bit period")

        # Wait for initial HS settle time
        await Timer(self.phy_config.t_hs_settle, units='ns')

        # Track consecutive invalid states to detect signal issues
        consecutive_invalid = 0
        max_consecutive_invalid = 10

        while self.hs_active:
            try:
                # Wait to sample at optimal point in bit period
                await Timer(sample_delay, units='ns')

                # Read differential data with error handling
                try:
                    p_val = int(self.sig_p.value)
                    n_val = int(self.sig_n.value)
                except (ValueError, TypeError):
                    # Handle 'z', 'x', or other invalid values
                    current_time = cocotb.utils.get_sim_time('ns')
                    self.signal_monitor.record_invalid_state(-1, -1, current_time)
                    consecutive_invalid += 1

                    if consecutive_invalid >= max_consecutive_invalid:
                        self.logger.warning(f"Lane {self.lane_index}: Too many consecutive invalid states, stopping HS monitoring")
                        break

                    self.logger.debug(f"Lane {self.lane_index}: Invalid signal values, skipping bit")
                    await Timer(bit_period - sample_delay, units='ns')
                    continue

                # Reset invalid counter on valid state
                consecutive_invalid = 0

                # HS-0: p=0, n=1; HS-1: p=1, n=0
                if p_val == 0 and n_val == 1:
                    current_time = cocotb.utils.get_sim_time('ns')
                    self.signal_monitor.record_hs_bit(False, current_time)
                    await self._handle_hs_bit(False)  # HS-0
                    self.logger.debug(f"Lane {self.lane_index}: Received HS-0 bit")
                elif p_val == 1 and n_val == 0:
                    current_time = cocotb.utils.get_sim_time('ns')
                    self.signal_monitor.record_hs_bit(True, current_time)
                    await self._handle_hs_bit(True)   # HS-1
                    self.logger.debug(f"Lane {self.lane_index}: Received HS-1 bit")
                else:
                    # Invalid HS state - log and skip
                    current_time = cocotb.utils.get_sim_time('ns')
                    self.signal_monitor.record_invalid_state(p_val, n_val, current_time)
                    consecutive_invalid += 1

                    if consecutive_invalid >= max_consecutive_invalid:
                        self.logger.warning(f"Lane {self.lane_index}: Too many consecutive invalid HS states, stopping HS monitoring")
                        break

                    self.logger.warning(f"Lane {self.lane_index}: Invalid HS state p={p_val}, n={n_val}")

                # Wait for the remainder of the bit period
                await Timer(bit_period - sample_delay, units='ns')

            except Exception as e:
                self.logger.error(f"Error in HS data monitoring: {e}")
                break

        self.logger.info(f"Lane {self.lane_index}: HS data monitoring ended")

    async def _handle_hs_prepare(self):
        """Handle HS prepare sequence detection"""
        self.logger.info(f"Lane {self.lane_index}: HS prepare detected, waiting for HS settle")

        # Wait for HS zero period and settle time
        await Timer(self.phy_config.t_hs_settle, units='ns')

        # Check for HS state more robustly using HS state decoder
        hs_detected = False
        for _ in range(5):  # Check multiple times to ensure stability
            current_hs_state = self._decode_hs_state()
            if current_hs_state in [DPhyState.HS_0, DPhyState.HS_1]:
                hs_detected = True
                self.logger.info(f"Lane {self.lane_index}: HS state detected: {current_hs_state}")
                break
            await Timer(2, units='ns')  # Small delay between checks

        if hs_detected:
            self.hs_active = True
            self.lp_active = False
            self.byte_buffer = 0
            self.bit_count = 0

            self.logger.info(f"Lane {self.lane_index}: Entered HS mode, starting data monitoring")

            # Start HS data monitoring
            if self._hs_monitor_task is None or self._hs_monitor_task.done():
                self._hs_monitor_task = cocotb.start_soon(self._monitor_hs_data())

            if self.on_hs_start:
                await self.on_hs_start()
        else:
            current_lp_state = self._decode_lp_state()
            self.logger.warning(f"Lane {self.lane_index}: Expected HS state but got LP state {current_lp_state}, staying in LP mode")

    async def _handle_hs_bit(self, bit_value: bool):
        """Handle received HS data bit"""
        # Accumulate bits into bytes (LSB first)
        if bit_value:
            self.byte_buffer |= (1 << self.bit_count)

        self.bit_count += 1

        if self.bit_count >= 8:
            # Complete byte received
            self.received_data.append(self.byte_buffer)
            self.logger.debug(f"Lane {self.lane_index}: Received byte 0x{self.byte_buffer:02x}")

            # Send chunk if buffer is full
            if len(self.received_data) >= self.chunk_size and self.on_data_received:
                chunk = bytes(self.received_data)
                self.logger.info(f"Lane {self.lane_index}: Sending {len(chunk)} bytes via callback")
                await self.on_data_received(chunk)
                self.received_data.clear()

            self.byte_buffer = 0
            self.bit_count = 0

    async def _handle_hs_exit(self):
        """Handle HS exit sequence"""
        self.logger.debug(f"Lane {self.lane_index}: HS exit detected")

        # Send any remaining buffered data
        if self.received_data and self.on_data_received:
            chunk = bytes(self.received_data)
            await self.on_data_received(chunk)
            self.received_data.clear()

        self.hs_active = False
        self.lp_active = True

        if self._hs_monitor_task and not self._hs_monitor_task.done():
            self._hs_monitor_task.kill()
        self._hs_monitor_task = None

        if self.on_hs_end:
            await self.on_hs_end()

    def get_received_data(self) -> bytes:
        """Get all received data and clear buffer"""
        # This method is no longer used for primary data retrieval
        # but is kept for compatibility. Data is now sent via callbacks.
        return b''

    def clear_received_data(self):
        """Clear received data buffer"""
        self.received_data.clear()
        self.byte_buffer = 0
        self.bit_count = 0

    def get_signal_statistics(self) -> dict:
        """Get signal quality statistics"""
        return self.signal_monitor.get_statistics()

    def reset_signal_monitor(self):
        """Reset signal quality monitoring"""
        self.signal_monitor.reset()


class DPhySignalMonitor:
    """D-PHY Signal Quality Monitor"""

    def __init__(self, lane_index: int):
        self.lane_index = lane_index
        self.signal_transitions = 0
        self.hs_bits_received = 0
        self.hs_bits_expected = 0
        self.invalid_states = 0
        self.timing_violations = 0
        self.last_transition_time = 0
        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.monitor.lane{lane_index}')

    def record_signal_transition(self, old_state: int, new_state: int, time_ns: int):
        """Record a signal state transition"""
        self.signal_transitions += 1
        self.last_transition_time = time_ns
        self.logger.debug(f"Lane {self.lane_index}: Signal transition {old_state}->{new_state} at {time_ns}ns")

    def record_hs_bit(self, bit_value: bool, time_ns: int):
        """Record a received HS bit"""
        self.hs_bits_received += 1
        self.logger.debug(f"Lane {self.lane_index}: HS bit {bit_value} at {time_ns}ns")

    def record_invalid_state(self, p_val: int, n_val: int, time_ns: int):
        """Record an invalid signal state"""
        self.invalid_states += 1
        self.logger.warning(f"Lane {self.lane_index}: Invalid state p={p_val}, n={n_val} at {time_ns}ns")

    def get_statistics(self) -> dict:
        """Get signal quality statistics"""
        return {
            'signal_transitions': self.signal_transitions,
            'hs_bits_received': self.hs_bits_received,
            'hs_bits_expected': self.hs_bits_expected,
            'invalid_states': self.invalid_states,
            'timing_violations': self.timing_violations,
            'bit_error_rate': (self.invalid_states / max(1, self.hs_bits_received)) * 100
        }

    def reset(self):
        """Reset all counters"""
        self.signal_transitions = 0
        self.hs_bits_received = 0
        self.hs_bits_expected = 0
        self.invalid_states = 0
        self.timing_violations = 0
        self.last_transition_time = 0


class DPhyModel:
    """Complete D-PHY model with clock and data lanes"""

    def __init__(self, bus: Csi2DPhyBus, config: Csi2Config):
        """
        Initialize complete D-PHY model

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

        # Create transmitters and receivers
        self.tx_lanes = []
        self.rx_lanes = []

        # Clock lane (index 0)
        self.clock_tx = DPhyTx(bus, config, self.phy_config, 0)
        self.clock_rx = DPhyRx(bus, config, self.phy_config, 0)

        # Data lanes (index 1+)
        for lane_idx in range(1, config.lane_count + 1):
            tx = DPhyTx(bus, config, self.phy_config, lane_idx)
            rx = DPhyRx(bus, config, self.phy_config, lane_idx)
            self.tx_lanes.append(tx)
            self.rx_lanes.append(rx)

        self.logger = logging.getLogger('cocotbext.csi2.dphy.model')

        # Clock generation for continuous clock mode
        if config.continuous_clock:
            self._clock_task = cocotb.start_soon(self._generate_continuous_clock())

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
        if not self.tx_lanes:
            raise Csi2PhyError("No data lanes configured")

        # Distribute data across lanes
        from ..utils import bytes_to_lanes
        lane_data = bytes_to_lanes(data, len(self.tx_lanes))

        # Send data on all lanes sequentially (cocotb compatibility)
        for tx_lane, lane_bytes in zip(self.tx_lanes, lane_data):
            await tx_lane.send_hs_data(lane_bytes)

    async def start_packet_transmission(self):
        """Start packet transmission (HS prepare on all lanes)"""
        if not self.config.continuous_clock:
            await self.clock_tx.start_hs_transmission()

        # Start all data lanes sequentially (cocotb compatibility)
        for tx_lane in self.tx_lanes:
            await tx_lane.start_hs_transmission()

    async def stop_packet_transmission(self):
        """Stop packet transmission (HS exit on all lanes)"""
        # Stop all data lanes sequentially (cocotb compatibility)
        for tx_lane in self.tx_lanes:
            await tx_lane.stop_hs_transmission()

        if not self.config.continuous_clock:
            await self.clock_tx.stop_hs_transmission()

    def get_received_data(self) -> bytes:
        """Collect received data from all RX lanes"""
        if not self.rx_lanes:
            return b''

        # Collect data from all lanes
        lane_data = []
        for rx_lane in self.rx_lanes:
            lane_data.append(rx_lane.get_received_data())

        # Merge lane data back into single stream
        from ..utils import lanes_to_bytes
        return lanes_to_bytes(lane_data)

    def set_rx_callbacks(self, on_packet_start=None, on_packet_end=None,
                        on_data_received=None):
        """Set callbacks for RX events"""
        # Set callbacks for all RX lanes
        for rx_lane in self.rx_lanes:
            rx_lane.on_hs_start = on_packet_start
            rx_lane.on_hs_end = on_packet_end
            rx_lane.on_data_received = on_data_received

    def get_signal_statistics(self) -> dict:
        """Get signal quality statistics from all lanes"""
        stats = {
            'clock_lane': self.clock_rx.get_signal_statistics(),
            'data_lanes': []
        }

        for rx_lane in self.rx_lanes:
            stats['data_lanes'].append(rx_lane.get_signal_statistics())

        return stats

    def reset_signal_monitors(self):
        """Reset signal quality monitoring on all lanes"""
        self.clock_rx.reset_signal_monitor()
        for rx_lane in self.rx_lanes:
            rx_lane.reset_signal_monitor()
