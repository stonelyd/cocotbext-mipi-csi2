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
from cocotb.triggers import Timer, RisingEdge, FallingEdge, Edge
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
        self.logger.debug(f"Lane {self.lane_index}: Starting HS prepare")
        
        # LP-00 (HS prepare)
        self._set_lp_state(DPhyState.LP_00)
        await Timer(self.phy_config.t_hs_prepare, units='ns')
        
        # HS-0 (HS zero)
        self.sig_p.value = 0
        self.sig_n.value = 1
        self.current_state = DPhyState.HS_0
        self.hs_active = True
        self.lp_active = False
        await Timer(self.phy_config.t_hs_zero, units='ns')
    
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
        
        for byte_val in data:
            await self._send_hs_byte(byte_val)
    
    async def _send_hs_byte(self, byte_val: int):
        """Send single byte in HS mode"""
        for bit_pos in range(8):
            bit_val = (byte_val >> bit_pos) & 1
            
            if bit_val:
                self.sig_p.value = 1
                self.sig_n.value = 0
                self.current_state = DPhyState.HS_1
            else:
                self.sig_p.value = 0
                self.sig_n.value = 1
                self.current_state = DPhyState.HS_0
            
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
        
        # Signal handles
        if self.is_clock_lane:
            self.sig_p = getattr(bus, 'clk_p', None)
            self.sig_n = getattr(bus, 'clk_n', None)
        else:
            self.sig_p = getattr(bus, f'data{lane_index-1}_p', None)
            self.sig_n = getattr(bus, f'data{lane_index-1}_n', None)
            
        if not self.sig_p or not self.sig_n:
            raise Csi2PhyError(f"Missing D-PHY signals for lane {lane_index}")
        
        # State tracking
        self.current_state = DPhyState.LP_11
        self.hs_active = False
        self.lp_active = True
        
        # Data capture
        self.received_data = bytearray()
        self.byte_buffer = 0
        self.bit_count = 0
        
        # Callbacks
        self.on_hs_start: Optional[Callable] = None
        self.on_hs_end: Optional[Callable] = None
        self.on_data_received: Optional[Callable] = None
        self.on_lp_sequence: Optional[Callable] = None
        
        self.logger = logging.getLogger(f'cocotbext.csi2.dphy.rx.lane{lane_index}')
        
        # Start monitoring
        self._monitor_task = cocotb.start_soon(self._monitor_signals())
    
    def _decode_lp_state(self) -> int:
        """Decode current LP state from signals"""
        p_val = int(self.sig_p.value)
        n_val = int(self.sig_n.value)  
        
        if p_val == 0 and n_val == 0:
            return DPhyState.LP_00
        elif p_val == 0 and n_val == 1:
            return DPhyState.LP_01
        elif p_val == 1 and n_val == 0:
            return DPhyState.LP_10
        else:  # p_val == 1 and n_val == 1
            return DPhyState.LP_11
    
    async def _monitor_signals(self):
        """Monitor D-PHY signals for state changes"""
        while True:
            try:
                # Wait for any signal change
                await Edge(self.sig_p)
                await Timer(1, units='ns')  # Small delay for signal stability
                
                new_state = self._decode_lp_state()
                
                if new_state != self.current_state:
                    await self._process_state_change(self.current_state, new_state)
                    self.current_state = new_state
                    
            except Exception as e:
                self.logger.error(f"Error in signal monitoring: {e}")
                break
    
    async def _process_state_change(self, old_state: int, new_state: int):
        """Process D-PHY state transitions"""
        self.logger.debug(f"Lane {self.lane_index}: State {old_state} -> {new_state}")
        
        # Detect HS prepare (LP-11 -> LP-00)
        if old_state == DPhyState.LP_11 and new_state == DPhyState.LP_00:
            await self._handle_hs_prepare()
        
        # Detect HS data (differential signaling)
        elif self.hs_active and new_state in [DPhyState.HS_0, DPhyState.HS_1]:
            await self._handle_hs_bit(new_state == DPhyState.HS_1)
        
        # Detect HS exit (back to LP-11)
        elif self.hs_active and new_state == DPhyState.LP_11:
            await self._handle_hs_exit()
        
        # LP sequence detection
        elif self.lp_active:
            if self.on_lp_sequence:
                await self.on_lp_sequence(new_state)
    
    async def _handle_hs_prepare(self):
        """Handle HS prepare sequence detection"""
        self.logger.debug(f"Lane {self.lane_index}: HS prepare detected")
        
        # Wait for HS zero period
        await Timer(self.phy_config.t_hs_settle, units='ns')
        
        self.hs_active = True
        self.lp_active = False
        self.byte_buffer = 0
        self.bit_count = 0
        
        if self.on_hs_start:
            await self.on_hs_start()
    
    async def _handle_hs_bit(self, bit_value: bool):
        """Handle received HS data bit"""
        # Accumulate bits into bytes (LSB first)
        if bit_value:
            self.byte_buffer |= (1 << self.bit_count)
        
        self.bit_count += 1
        
        if self.bit_count >= 8:
            # Complete byte received
            self.received_data.append(self.byte_buffer)
            
            if self.on_data_received:
                await self.on_data_received(self.byte_buffer)
            
            self.byte_buffer = 0
            self.bit_count = 0
    
    async def _handle_hs_exit(self):
        """Handle HS exit sequence"""
        self.logger.debug(f"Lane {self.lane_index}: HS exit detected")
        
        self.hs_active = False
        self.lp_active = True
        
        if self.on_hs_end:
            await self.on_hs_end()
    
    def get_received_data(self) -> bytes:
        """Get all received data and clear buffer"""
        data = bytes(self.received_data)
        self.received_data.clear()
        return data
    
    def clear_received_data(self):
        """Clear received data buffer"""
        self.received_data.clear()
        self.byte_buffer = 0
        self.bit_count = 0


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
        self.phy_config = Csi2PhyConfig()
        
        # Validate timing parameters
        if not self.phy_config.validate_timing(config.bit_rate_mbps):
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
        
        # Send data on all lanes simultaneously
        send_tasks = []
        for tx_lane, lane_bytes in zip(self.tx_lanes, lane_data):
            task = cocotb.start_soon(tx_lane.send_hs_data(lane_bytes))
            send_tasks.append(task)
        
        # Wait for all lanes to complete
        await asyncio.gather(*send_tasks)
    
    async def start_packet_transmission(self):
        """Start packet transmission (HS prepare on all lanes)"""
        if not self.config.continuous_clock:
            await self.clock_tx.start_hs_transmission()
        
        # Start all data lanes
        start_tasks = []
        for tx_lane in self.tx_lanes:
            task = cocotb.start_soon(tx_lane.start_hs_transmission())
            start_tasks.append(task)
        
        await asyncio.gather(*start_tasks)
    
    async def stop_packet_transmission(self):
        """Stop packet transmission (HS exit on all lanes)"""
        # Stop all data lanes
        stop_tasks = []
        for tx_lane in self.tx_lanes:
            task = cocotb.start_soon(tx_lane.stop_hs_transmission())
            stop_tasks.append(task)
        
        await asyncio.gather(*stop_tasks)
        
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
        for rx_lane in self.rx_lanes:
            if on_packet_start:
                rx_lane.on_hs_start = on_packet_start
            if on_packet_end:
                rx_lane.on_hs_end = on_packet_end
            if on_data_received:
                rx_lane.on_data_received = on_data_received
