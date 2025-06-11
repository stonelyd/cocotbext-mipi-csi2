"""
MIPI C-PHY Physical Layer Model

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
from typing import List, Optional, Callable, Tuple
import logging
from ..config import Csi2Config, Csi2PhyConfig
from ..bus import Csi2CPhyBus
from ..exceptions import Csi2PhyError, Csi2TimingError


class CPhyState:
    """C-PHY 3-Phase States"""
    # C-PHY uses 3-phase encoding with states 0-6
    STATE_0 = 0  # ABC = 001
    STATE_1 = 1  # ABC = 010  
    STATE_2 = 2  # ABC = 100
    STATE_3 = 3  # ABC = 110
    STATE_4 = 4  # ABC = 101
    STATE_5 = 5  # ABC = 011
    STATE_6 = 6  # ABC = 000
    
    # Low Power states
    LP_000 = 7   # All low
    LP_111 = 8   # All high


class CPhySymbolEncoder:
    """C-PHY Symbol Encoder/Decoder"""
    
    # C-PHY encoding table (simplified)
    # Maps 7-bit data to 3 symbols (each symbol is 3 bits representing state transition)
    ENCODE_TABLE = {
        # This is a simplified encoding table
        # Full C-PHY encoding is much more complex
        0x00: [(0, 1), (2, 3), (4, 5)],
        0x01: [(1, 2), (3, 4), (5, 0)],
        0x02: [(2, 3), (4, 5), (0, 1)],
        # ... more entries would be needed for full implementation
    }
    
    @classmethod
    def encode_byte(cls, byte_val: int) -> List[Tuple[int, int]]:
        """
        Encode byte into C-PHY state transitions
        
        Args:
            byte_val: 8-bit value to encode
            
        Returns:
            List of (current_state, next_state) tuples
        """
        # Simplified encoding - maps byte to state transitions
        # Real C-PHY encoding is much more sophisticated
        transitions = []
        
        # Split byte into symbols
        symbol1 = byte_val & 0x07
        symbol2 = (byte_val >> 3) & 0x07
        symbol3 = (byte_val >> 6) & 0x03
        
        # Convert symbols to state transitions
        transitions.append((symbol1, (symbol1 + 1) % 7))
        transitions.append((symbol2, (symbol2 + 2) % 7))  
        transitions.append((symbol3, (symbol3 + 3) % 7))
        
        return transitions
    
    @classmethod
    def decode_transitions(cls, transitions: List[Tuple[int, int]]) -> int:
        """
        Decode C-PHY state transitions back to byte
        
        Args:
            transitions: List of (current_state, next_state) tuples
            
        Returns:
            Decoded byte value
        """
        if len(transitions) != 3:
            raise ValueError("C-PHY requires 3 transitions per byte")
        
        # Simplified decoding
        symbol1 = transitions[0][0]
        symbol2 = transitions[1][0] 
        symbol3 = transitions[2][0]
        
        byte_val = symbol1 | (symbol2 << 3) | (symbol3 << 6)
        return byte_val & 0xFF


class CPhyTx:
    """C-PHY Transmitter Model"""
    
    def __init__(self, bus: Csi2CPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, trio_index: int = 0):
        """
        Initialize C-PHY transmitter
        
        Args:
            bus: C-PHY bus interface
            config: CSI-2 configuration  
            phy_config: PHY-specific configuration
            trio_index: Trio index (0, 1, 2)
        """
        self.bus = bus
        self.config = config
        self.phy_config = phy_config
        self.trio_index = trio_index
        
        # Signal handles
        self.sig_a = getattr(bus, f'trio{trio_index}_a', None)
        self.sig_b = getattr(bus, f'trio{trio_index}_b', None)
        self.sig_c = getattr(bus, f'trio{trio_index}_c', None)
        
        if not all([self.sig_a, self.sig_b, self.sig_c]):
            raise Csi2PhyError(f"Missing C-PHY signals for trio {trio_index}")
        
        # State tracking
        self.current_state = CPhyState.LP_111
        self.hs_active = False
        self.encoder = CPhySymbolEncoder()
        
        # Timing
        self.symbol_period_ns = config.get_bit_period_ns() * 3  # 3 UI per symbol
        
        # Initialize to LP state
        self._set_lp_state(CPhyState.LP_111)
        
        self.logger = logging.getLogger(f'cocotbext.csi2.cphy.tx.trio{trio_index}')
    
    def _set_lp_state(self, state: int):
        """Set Low Power state on trio signals"""
        self.current_state = state
        
        if state == CPhyState.LP_000:
            self.sig_a.value = 0
            self.sig_b.value = 0
            self.sig_c.value = 0
        elif state == CPhyState.LP_111:
            self.sig_a.value = 1
            self.sig_b.value = 1
            self.sig_c.value = 1
    
    def _set_hs_state(self, state: int):
        """Set High Speed 3-phase state"""
        self.current_state = state
        
        # C-PHY state to signal mapping
        state_map = {
            CPhyState.STATE_0: (0, 0, 1),  # ABC = 001
            CPhyState.STATE_1: (0, 1, 0),  # ABC = 010
            CPhyState.STATE_2: (1, 0, 0),  # ABC = 100
            CPhyState.STATE_3: (1, 1, 0),  # ABC = 110
            CPhyState.STATE_4: (1, 0, 1),  # ABC = 101
            CPhyState.STATE_5: (0, 1, 1),  # ABC = 011
            CPhyState.STATE_6: (0, 0, 0),  # ABC = 000
        }
        
        if state in state_map:
            a, b, c = state_map[state]
            self.sig_a.value = a
            self.sig_b.value = b
            self.sig_c.value = c
    
    async def _hs_prepare_sequence(self):
        """Execute HS prepare sequence for C-PHY"""
        self.logger.debug(f"Trio {self.trio_index}: Starting HS prepare")
        
        # LP-000 (prepare phase)
        self._set_lp_state(CPhyState.LP_000)
        await Timer(self.phy_config.t_3phase_prepare, units='ns')
        
        # Enter HS mode with initial state
        self._set_hs_state(CPhyState.STATE_0)
        self.hs_active = True
        await Timer(self.phy_config.t_3phase_zero, units='ns')
    
    async def _hs_exit_sequence(self):
        """Execute HS exit sequence"""
        self.logger.debug(f"Trio {self.trio_index}: Starting HS exit")
        
        # Return to LP-111
        self._set_lp_state(CPhyState.LP_111)
        self.hs_active = False
        await Timer(self.phy_config.t_3phase_post, units='ns')
    
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
        """Send single byte using C-PHY encoding"""
        # Encode byte to state transitions
        transitions = self.encoder.encode_byte(byte_val)
        
        # Send each transition
        for current_state, next_state in transitions:
            self._set_hs_state(current_state)
            await Timer(self.symbol_period_ns / 2, units='ns')
            self._set_hs_state(next_state)
            await Timer(self.symbol_period_ns / 2, units='ns')
    
    async def start_hs_transmission(self):
        """Start High Speed transmission mode"""
        if not self.hs_active:
            await self._hs_prepare_sequence()
    
    async def stop_hs_transmission(self):
        """Stop High Speed transmission mode"""
        if self.hs_active:
            await self._hs_exit_sequence()


class CPhyRx:
    """C-PHY Receiver Model"""
    
    def __init__(self, bus: Csi2CPhyBus, config: Csi2Config,
                 phy_config: Csi2PhyConfig, trio_index: int = 0):
        """
        Initialize C-PHY receiver
        
        Args:
            bus: C-PHY bus interface
            config: CSI-2 configuration
            phy_config: PHY-specific configuration  
            trio_index: Trio index (0, 1, 2)
        """
        self.bus = bus
        self.config = config
        self.phy_config = phy_config
        self.trio_index = trio_index
        
        # Signal handles
        self.sig_a = getattr(bus, f'trio{trio_index}_a', None)
        self.sig_b = getattr(bus, f'trio{trio_index}_b', None)
        self.sig_c = getattr(bus, f'trio{trio_index}_c', None)
        
        if not all([self.sig_a, self.sig_b, self.sig_c]):
            raise Csi2PhyError(f"Missing C-PHY signals for trio {trio_index}")
        
        # State tracking
        self.current_state = CPhyState.LP_111
        self.hs_active = False
        self.decoder = CPhySymbolEncoder()
        
        # Data capture
        self.received_data = bytearray()
        self.symbol_buffer = []
        
        # Callbacks
        self.on_hs_start: Optional[Callable] = None
        self.on_hs_end: Optional[Callable] = None
        self.on_data_received: Optional[Callable] = None
        
        self.logger = logging.getLogger(f'cocotbext.csi2.cphy.rx.trio{trio_index}')
        
        # Start monitoring
        self._monitor_task = cocotb.start_soon(self._monitor_signals())
    
    def _decode_current_state(self) -> int:
        """Decode current state from trio signals"""
        a = int(self.sig_a.value)
        b = int(self.sig_b.value)
        c = int(self.sig_c.value)
        
        # Map signal combination to state
        signal_map = {
            (0, 0, 1): CPhyState.STATE_0,
            (0, 1, 0): CPhyState.STATE_1,
            (1, 0, 0): CPhyState.STATE_2,
            (1, 1, 0): CPhyState.STATE_3,
            (1, 0, 1): CPhyState.STATE_4,
            (0, 1, 1): CPhyState.STATE_5,
            (0, 0, 0): CPhyState.LP_000,
            (1, 1, 1): CPhyState.LP_111,
        }
        
        return signal_map.get((a, b, c), CPhyState.LP_111)
    
    async def _monitor_signals(self):
        """Monitor C-PHY signals for state changes"""
        while True:
            try:
                # Wait for any signal change
                await Edge(self.sig_a)
                await Timer(1, units='ns')  # Stability delay
                
                new_state = self._decode_current_state()
                
                if new_state != self.current_state:
                    await self._process_state_change(self.current_state, new_state)
                    self.current_state = new_state
                    
            except Exception as e:
                self.logger.error(f"Error in signal monitoring: {e}")
                break
    
    async def _process_state_change(self, old_state: int, new_state: int):
        """Process C-PHY state transitions"""
        self.logger.debug(f"Trio {self.trio_index}: State {old_state} -> {new_state}")
        
        # Detect HS prepare (LP-111 -> LP-000)
        if old_state == CPhyState.LP_111 and new_state == CPhyState.LP_000:
            await self._handle_hs_prepare()
        
        # Detect HS data (3-phase states)
        elif self.hs_active and new_state in range(7):  # States 0-6
            await self._handle_hs_transition(old_state, new_state)
        
        # Detect HS exit (back to LP-111) 
        elif self.hs_active and new_state == CPhyState.LP_111:
            await self._handle_hs_exit()
    
    async def _handle_hs_prepare(self):
        """Handle HS prepare sequence detection"""
        self.logger.debug(f"Trio {self.trio_index}: HS prepare detected")
        
        self.hs_active = True
        self.symbol_buffer.clear()
        
        if self.on_hs_start:
            await self.on_hs_start()
    
    async def _handle_hs_transition(self, old_state: int, new_state: int):
        """Handle HS state transition"""
        # Store transition for decoding
        self.symbol_buffer.append((old_state, new_state))
        
        # C-PHY sends 3 transitions per byte
        if len(self.symbol_buffer) >= 3:
            try:
                # Decode transitions to byte
                byte_val = self.decoder.decode_transitions(self.symbol_buffer[-3:])
                self.received_data.append(byte_val)
                
                if self.on_data_received:
                    await self.on_data_received(byte_val)
                    
            except Exception as e:
                self.logger.warning(f"Failed to decode transitions: {e}")
    
    async def _handle_hs_exit(self):
        """Handle HS exit sequence"""
        self.logger.debug(f"Trio {self.trio_index}: HS exit detected")
        
        self.hs_active = False
        
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
        self.symbol_buffer.clear()


class CPhyModel:
    """Complete C-PHY model with multiple trios"""
    
    def __init__(self, bus: Csi2CPhyBus, config: Csi2Config):
        """
        Initialize complete C-PHY model
        
        Args:
            bus: C-PHY bus interface
            config: CSI-2 configuration
        """
        self.bus = bus
        self.config = config
        self.phy_config = Csi2PhyConfig()
        
        # Create transmitters and receivers for each trio
        self.tx_trios = []
        self.rx_trios = []
        
        for trio_idx in range(config.trio_count):
            tx = CPhyTx(bus, config, self.phy_config, trio_idx)
            rx = CPhyRx(bus, config, self.phy_config, trio_idx)
            self.tx_trios.append(tx)
            self.rx_trios.append(rx)
        
        self.logger = logging.getLogger('cocotbext.csi2.cphy.model')
    
    async def send_packet_data(self, data: bytes):
        """
        Send packet data across all trios
        
        Args:
            data: Packet data to transmit
        """
        if not self.tx_trios:
            raise Csi2PhyError("No trios configured")
        
        # Distribute data across trios (similar to D-PHY lanes)
        from ..utils import bytes_to_lanes
        trio_data = bytes_to_lanes(data, len(self.tx_trios))
        
        # Send data on all trios simultaneously
        send_tasks = []
        for tx_trio, trio_bytes in zip(self.tx_trios, trio_data):
            task = cocotb.start_soon(tx_trio.send_hs_data(trio_bytes))
            send_tasks.append(task)
        
        await asyncio.gather(*send_tasks)
    
    async def start_packet_transmission(self):
        """Start packet transmission on all trios"""
        start_tasks = []
        for tx_trio in self.tx_trios:
            task = cocotb.start_soon(tx_trio.start_hs_transmission())
            start_tasks.append(task)
        
        await asyncio.gather(*start_tasks)
    
    async def stop_packet_transmission(self):
        """Stop packet transmission on all trios"""
        stop_tasks = []
        for tx_trio in self.tx_trios:
            task = cocotb.start_soon(tx_trio.stop_hs_transmission())
            stop_tasks.append(task)
        
        await asyncio.gather(*stop_tasks)
    
    def get_received_data(self) -> bytes:
        """Collect received data from all RX trios"""
        if not self.rx_trios:
            return b''
        
        # Collect data from all trios
        trio_data = []
        for rx_trio in self.rx_trios:
            trio_data.append(rx_trio.get_received_data())
        
        # Merge trio data back into single stream
        from ..utils import lanes_to_bytes
        return lanes_to_bytes(trio_data)
    
    def set_rx_callbacks(self, on_packet_start=None, on_packet_end=None,
                        on_data_received=None):
        """Set callbacks for RX events"""
        for rx_trio in self.rx_trios:
            if on_packet_start:
                rx_trio.on_hs_start = on_packet_start
            if on_packet_end:
                rx_trio.on_hs_end = on_packet_end
            if on_data_received:
                rx_trio.on_data_received = on_data_received
