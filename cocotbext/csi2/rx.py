"""
MIPI CSI-2 Receiver Model

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
from cocotb.triggers import Timer, Event, First, with_timeout
from cocotb.queue import Queue
import asyncio
import logging
from typing import List, Optional, Union, Callable, Dict
from collections import defaultdict
from .bus import Csi2Bus, Csi2DPhyBus, Csi2CPhyBus
from .config import Csi2Config, PhyType, DataType, VirtualChannel
from .packet import Csi2Packet, Csi2ShortPacket, Csi2LongPacket, Csi2PacketParser
from .phy import DPhyRxModel, CPhyModel
from .exceptions import Csi2Exception, Csi2ProtocolError, Csi2PhyError, Csi2TimingError
from .utils import setup_logging


class Csi2FrameBuffer:
    """Buffer for assembling complete image frames from CSI-2 packets"""

    def __init__(self, virtual_channel: int):
        self.virtual_channel = virtual_channel
        self.frame_number = 0
        self.frame_active = False
        self.line_active = False
        self.current_line = 0
        self.frame_data = bytearray()
        self.line_data = bytearray()
        self.frame_width = 0
        self.frame_height = 0
        self.data_type = None
        self.timestamp_start = None
        self.timestamp_end = None

    def start_frame(self, frame_number: int):
        """Start new frame"""
        self.frame_number = frame_number
        self.frame_active = True
        self.current_line = 1
        self.frame_data.clear()
        self.line_data.clear()
        self.line_active = False
        self.timestamp_start = cocotb.utils.get_sim_time('ns')

    def end_frame(self, frame_number: int):
        """End current frame"""
        if self.frame_number != frame_number:
            raise Csi2ProtocolError(f"Frame number mismatch: expected {self.frame_number}, got {frame_number}")

        self.frame_active = False
        self.timestamp_end = cocotb.utils.get_sim_time('ns')

    def start_line(self, line_number: int):
        """Start new line"""
        if not self.frame_active:
            raise Csi2ProtocolError("Line start without active frame")

        if self.current_line != line_number:
            raise Csi2ProtocolError(f"Line number mismatch: expected {self.current_line}, got {line_number}")

        self.line_active = True
        self.line_data.clear()

    def end_line(self, line_number: int):
        """End current line"""
        if self.current_line != line_number:
            raise Csi2ProtocolError(f"Line number mismatch: expected {self.current_line}, got {line_number}")

        # Add line data to frame
        self.frame_data.extend(self.line_data)
        self.line_active = False
        self.current_line += 1

    def add_pixel_data(self, data_type: DataType, payload: bytes):
        """Add pixel data to current line"""
        if not self.line_active:
            raise Csi2ProtocolError("Pixel data without active line")

        self.data_type = data_type
        self.line_data.extend(payload)

    def get_frame_data(self) -> bytes:
        """Get complete frame data"""
        return bytes(self.frame_data)

    def is_frame_complete(self) -> bool:
        """Check if frame is complete"""
        return not self.frame_active and len(self.frame_data) > 0

    def get_frame_info(self) -> dict:
        """Get frame information"""
        return {
            'frame_number': self.frame_number,
            'virtual_channel': self.virtual_channel,
            'data_type': self.data_type,
            'size_bytes': len(self.frame_data),
            'timestamp_start': self.timestamp_start,
            'timestamp_end': self.timestamp_end,
            'duration_ns': self.timestamp_end - self.timestamp_start if self.timestamp_end else None
        }

    def reset(self):
        """Reset frame buffer state"""
        self.frame_number = 0
        self.frame_active = False
        self.line_active = False
        self.current_line = 0
        self.frame_data.clear()
        self.line_data.clear()
        self.frame_width = 0
        self.frame_height = 0
        self.data_type = None
        self.timestamp_start = None
        self.timestamp_end = None


class Csi2RxModel:
    """
    CSI-2 Receiver Model for testing transmitter implementations

    This model can receive and validate CSI-2 packet streams with
    comprehensive error detection and frame assembly.
    """

    def __init__(self, bus: Union[Csi2Bus, Csi2DPhyBus, Csi2CPhyBus],
                 config: Csi2Config, phy_model=None):
        """
        Initialize CSI-2 receiver model

        Args:
            bus: CSI-2 bus interface (D-PHY or C-PHY)
            config: CSI-2 configuration
            phy_model: Optional shared PHY model (creates new one if None)
        """
        self.bus = bus
        self.config = config

        # Use provided PHY model or create new one
        if phy_model is not None:
            self.phy_model = phy_model
            # Determine bus type from PHY model
            if isinstance(phy_model, DPhyRxModel):
                self.phy_bus = phy_model.bus
            elif isinstance(phy_model, CPhyModel):
                self.phy_bus = phy_model.bus
            else:
                raise Csi2ProtocolError(f"Unsupported PHY model type: {type(phy_model)}")
        else:
            # Initialize PHY model based on configuration
            if config.phy_type == PhyType.DPHY:
                if not isinstance(bus, Csi2DPhyBus):
                    self.phy_bus = Csi2DPhyBus(bus.entity, config.lane_count)
                else:
                    self.phy_bus = bus
                self.phy_model = DPhyRxModel(self.phy_bus, config)
            elif config.phy_type == PhyType.CPHY:
                if not isinstance(bus, Csi2CPhyBus):
                    self.phy_bus = Csi2CPhyBus(bus.entity, config.trio_count)
                else:
                    self.phy_bus = bus
                self.phy_model = CPhyModel(self.phy_bus, config)
            else:
                raise Csi2ProtocolError(f"Unsupported PHY type: {config.phy_type}")

        # Packet parser
        self.parser = Csi2PacketParser()

        # Raw data buffer for accumulating bytes from PHY
        self.raw_data_buffer = bytearray()

        # Frame buffers for each virtual channel
        self.frame_buffers = {vc: Csi2FrameBuffer(vc) for vc in range(16)}

        # Received packet storage
        self.received_packets = Queue()
        self.completed_frames = Queue()

        # Statistics and monitoring
        self.packets_received = 0
        self.bytes_received = 0
        self.frames_received = 0
        self.ecc_errors = 0
        self.checksum_errors = 0
        self.protocol_errors = 0
        self.timing_violations = 0

        # Timing validation
        self.last_packet_time = 0
        self.frame_start_times = {}
        self.line_start_times = {}

        # Control flags
        self.receiving = False
        self.frame_assembly_enabled = True
        self.strict_timing_check = False

        # Events
        self.packet_received_event = Event()
        self.frame_complete_event = Event()
        self.error_detected_event = Event()

        # Callbacks
        self.on_packet_received: Optional[Callable] = None
        self.on_frame_complete: Optional[Callable] = None
        self.on_error_detected: Optional[Callable] = None
        self.on_frame_start: Optional[Callable] = None
        self.on_frame_end: Optional[Callable] = None
        self.on_line_start: Optional[Callable] = None
        self.on_line_end: Optional[Callable] = None

        self.logger = logging.getLogger('cocotbext.csi2.rx')

        # Set up PHY callbacks
        self.phy_model.set_rx_callbacks(
            on_packet_start=self._on_phy_packet_start,
            on_packet_end=self._on_phy_packet_end,
            on_data_received=self._on_phy_data_received
        )

        # Reception is handled by PHY callbacks
        # self._rx_task = cocotb.start_soon(self._reception_handler())

    async def _on_phy_packet_start(self):
        """Handle PHY packet start"""
        self.receiving = True
        self.logger.info("RX: PHY packet transmission started")

    async def _on_phy_packet_end(self):
        """Handle PHY packet end"""
        self.receiving = False
        self.logger.info("RX: PHY packet transmission ended")

        # Process buffered data
        if self.raw_data_buffer:
            raw_data = bytes(self.raw_data_buffer)
            self.logger.info(f"RX: Processing {len(raw_data)} bytes from buffer")
            await self._process_raw_data(raw_data)
            self.raw_data_buffer.clear()
            # Don't reset parser here - it should maintain state across packets

    async def _on_phy_data_received(self, byte_data):
        """Handle PHY data reception"""
        # Accumulate received bytes for processing
        if isinstance(byte_data, int):
            # Single byte received
            self.raw_data_buffer.append(byte_data)
        else:
            # Multiple bytes received
            self.raw_data_buffer.extend(byte_data)

    async def _process_raw_data(self, raw_data: bytes):
        """
        Process raw received data into packets

        Args:
            raw_data: Raw bytes from PHY layer
        """
        self.bytes_received += len(raw_data)

        # Debug: log raw data
        self.logger.info(f"RX: Raw data received: {[f'{b:02x}' for b in raw_data[:20]]}...")

        # Feed data to parser
        new_packets = self.parser.feed_data(raw_data)

        # Debug: log parsed packets
        for packet in new_packets:
            self.logger.info(f"RX: Parsed packet: VC={packet.virtual_channel}, "
                             f"DT=0x{packet.data_type:02x}, "
                             f"Type={'Short' if packet.is_short_packet() else 'Long'}")

        # Process each parsed packet
        for packet in new_packets:
            await self._process_packet(packet)

    async def _process_packet(self, packet: Csi2Packet):
        """
        Process a received CSI-2 packet

        Args:
            packet: Parsed CSI-2 packet
        """
        current_time = cocotb.utils.get_sim_time('ns')
        packet.timestamp = current_time

        # Update statistics
        self.packets_received += 1

        # Timing validation
        if self.strict_timing_check:
            await self._validate_packet_timing(packet, current_time)

        # Store packet
        await self.received_packets.put(packet)

        # Process based on packet type
        try:
            if isinstance(packet, Csi2ShortPacket):
                await self._process_short_packet(packet)
            elif isinstance(packet, Csi2LongPacket):
                await self._process_long_packet(packet)

        except Exception as e:
            self.protocol_errors += 1
            self.logger.error(f"Packet processing error: {e}")
            if self.on_error_detected:
                await self.on_error_detected('protocol', packet, str(e))

        # Trigger events and callbacks
        self.packet_received_event.set()
        if self.on_packet_received:
            await self.on_packet_received(packet)

        self.logger.debug(f"Processed packet: VC{packet.virtual_channel} "
                         f"DT{packet.data_type:02X}")

    async def _process_short_packet(self, packet: Csi2ShortPacket):
        """Process short packet (synchronization)"""
        vc = packet.virtual_channel
        frame_buffer = self.frame_buffers[vc]

        if packet.data_type == DataType.FRAME_START:
            frame_number = packet.data
            self.logger.info(f"Frame start: VC{vc} Frame#{frame_number}")

            if self.frame_assembly_enabled:
                frame_buffer.start_frame(frame_number)

            self.frame_start_times[vc] = packet.timestamp

            if self.on_frame_start:
                await self.on_frame_start(vc, frame_number)

        elif packet.data_type == DataType.FRAME_END:
            frame_number = packet.data
            self.logger.info(f"Frame end: VC{vc} Frame#{frame_number}")

            if self.frame_assembly_enabled:
                frame_buffer.end_frame(frame_number)

                # Frame is complete
                if frame_buffer.is_frame_complete():
                    await self.completed_frames.put(frame_buffer.get_frame_info())
                    self.frames_received += 1
                    self.frame_complete_event.set()

                    if self.on_frame_complete:
                        await self.on_frame_complete(frame_buffer)

            if self.on_frame_end:
                await self.on_frame_end(vc, frame_number)

        elif packet.data_type == DataType.LINE_START:
            line_number = packet.data
            self.logger.debug(f"Line start: VC{vc} Line#{line_number}")

            if self.frame_assembly_enabled:
                frame_buffer.start_line(line_number)

            self.line_start_times[f"{vc}_{line_number}"] = packet.timestamp

            if self.on_line_start:
                await self.on_line_start(vc, line_number)

        elif packet.data_type == DataType.LINE_END:
            line_number = packet.data
            self.logger.debug(f"Line end: VC{vc} Line#{line_number}")

            if self.frame_assembly_enabled:
                frame_buffer.end_line(line_number)

            if self.on_line_end:
                await self.on_line_end(vc, line_number)

    async def _process_long_packet(self, packet: Csi2LongPacket):
        """Process long packet (pixel data)"""
        vc = packet.virtual_channel

        # Validate checksum
        if not packet.validate_checksum():
            self.checksum_errors += 1
            self.logger.warning(f"Checksum error in VC{vc} packet")
            if self.on_error_detected:
                await self.on_error_detected('checksum', packet, "Invalid checksum")

        # Add to frame buffer if enabled
        if self.frame_assembly_enabled:
            frame_buffer = self.frame_buffers[vc]
            frame_buffer.add_pixel_data(DataType(packet.data_type), packet.payload)

    async def _validate_packet_timing(self, packet: Csi2Packet, current_time: int):
        """Validate packet timing constraints"""
        if self.last_packet_time > 0:
            inter_packet_time = current_time - self.last_packet_time
            min_spacing_ns = self.config.clock_period_ns * 4  # Minimum 4 clock cycles

            if inter_packet_time < min_spacing_ns:
                self.timing_violations += 1
                self.logger.warning(f"Timing violation: inter-packet spacing "
                                  f"{inter_packet_time}ns < {min_spacing_ns}ns")
                if self.on_error_detected:
                    await self.on_error_detected('timing', packet,
                                                f"Inter-packet spacing violation")

        self.last_packet_time = current_time

    async def get_next_packet(self, timeout_ns: Optional[int] = None) -> Optional[Csi2Packet]:
        """
        Get next received packet

        Args:
            timeout_ns: Timeout in nanoseconds (None for no timeout)

        Returns:
            Next packet or None if timeout
        """
        if timeout_ns:
            try:
                return await with_timeout(self.received_packets.get(), timeout_ns, 'ns')
            except cocotb.result.SimTimeoutError:
                return None
        else:
            if self.received_packets.empty():
                return None
            return await self.received_packets.get()

    async def get_next_frame(self, timeout_ns: Optional[int] = None) -> Optional[dict]:
        """
        Get next completed frame

        Args:
            timeout_ns: Timeout in nanoseconds (None for no timeout)

        Returns:
            Frame info dictionary or None if timeout
        """
        if timeout_ns:
            try:
                return await with_timeout(self.completed_frames.get(), timeout_ns, 'ns')
            except cocotb.result.SimTimeoutError:
                return None
        else:
            return await self.completed_frames.get()

    def get_received_packets(self, virtual_channel: Optional[int] = None,
                           data_type: Optional[DataType] = None) -> List[Csi2Packet]:
        """
        Get all received packets with optional filtering

        Args:
            virtual_channel: Filter by virtual channel
            data_type: Filter by data type

        Returns:
            List of matching packets
        """
        packets = []
        temp_queue = Queue()

        # Extract all packets
        while not self.received_packets.empty():
            packet = self.received_packets.get_nowait()

            # Apply filters
            if virtual_channel is not None and packet.virtual_channel != virtual_channel:
                temp_queue.put_nowait(packet)
                continue

            if data_type is not None and packet.data_type != data_type.value:
                temp_queue.put_nowait(packet)
                continue

            packets.append(packet)

        # Put back non-matching packets
        while not temp_queue.empty():
            self.received_packets.put_nowait(temp_queue.get_nowait())

        return packets

    def get_frame_data(self, virtual_channel: int) -> Optional[bytes]:
        """
        Get assembled frame data for virtual channel

        Args:
            virtual_channel: Virtual channel ID

        Returns:
            Frame data or None if no complete frame
        """
        frame_buffer = self.frame_buffers[virtual_channel]
        if frame_buffer.is_frame_complete():
            return frame_buffer.get_frame_data()
        return None

    def enable_strict_timing_validation(self, enable: bool = True):
        """
        Enable/disable strict timing validation

        Args:
            enable: Enable strict timing checks
        """
        self.strict_timing_check = enable
        self.logger.info(f"Strict timing validation: {'enabled' if enable else 'disabled'}")

    def enable_frame_assembly(self, enable: bool = True):
        """
        Enable/disable frame assembly

        Args:
            enable: Enable frame assembly
        """
        self.frame_assembly_enabled = enable
        self.logger.info(f"Frame assembly: {'enabled' if enable else 'disabled'}")

    def get_statistics(self) -> dict:
        """Get reception statistics"""
        return {
            'packets_received': self.packets_received,
            'bytes_received': self.bytes_received,
            'frames_received': self.frames_received,
            'ecc_errors': self.ecc_errors + self.parser.ecc_error_count,
            'checksum_errors': self.checksum_errors + self.parser.checksum_error_count,
            'protocol_errors': self.protocol_errors,
            'timing_violations': self.timing_violations,
            'parser_errors': self.parser.error_count,
            'buffer_size': self.parser.packet_buffer.__len__(),
            'queue_sizes': {
                'packets': self.received_packets.qsize(),
                'frames': self.completed_frames.qsize()
            },
            'receiving': self.receiving
        }

    def get_frame_statistics(self) -> Dict[int, dict]:
        """Get per-VC frame statistics"""
        stats = {}
        for vc, buffer in self.frame_buffers.items():
            stats[vc] = {
                'frame_active': buffer.frame_active,
                'line_active': buffer.line_active,
                'current_line': buffer.current_line,
                'frame_number': buffer.frame_number,
                'data_size': len(buffer.frame_data),
                'data_type': buffer.data_type.value if buffer.data_type else None
            }
        return stats

    def reset_statistics(self):
        """Reset reception statistics"""
        self.packets_received = 0
        self.bytes_received = 0
        self.frames_received = 0
        self.ecc_errors = 0
        self.checksum_errors = 0
        self.protocol_errors = 0
        self.timing_violations = 0
        self.parser.reset()

    def clear_buffers(self):
        """Clear all buffers and queues"""
        # Clear packet queue
        while not self.received_packets.empty():
            self.received_packets.get_nowait()

        # Clear frame queue
        while not self.completed_frames.empty():
            self.completed_frames.get_nowait()

        # Clear raw data buffer
        self.raw_data_buffer.clear()

        # Reset packet parser
        self.parser.reset()

        # Reset frame buffers
        for vc in range(16):
            self.frame_buffers[vc] = Csi2FrameBuffer(vc)

        self.logger.info("All buffers cleared")

    async def wait_for_frame(self, virtual_channel: int = 0,
                           timeout_ns: int = 1000000) -> bool:
        """
        Wait for a complete frame on specified virtual channel

        Args:
            virtual_channel: Virtual channel to monitor
            timeout_ns: Timeout in nanoseconds

        Returns:
            True if frame received, False if timeout
        """
        start_time = cocotb.utils.get_sim_time('ns')

        while True:
            frame_buffer = self.frame_buffers[virtual_channel]
            if frame_buffer.is_frame_complete():
                return True

            current_time = cocotb.utils.get_sim_time('ns')
            if current_time - start_time > timeout_ns:
                return False

            await Timer(1000, units='ns')  # Check every 1us

    async def wait_for_packets(self, count: int, timeout_ns: int = 1000000) -> bool:
        """
        Wait for specified number of packets

        Args:
            count: Number of packets to wait for
            timeout_ns: Timeout in nanoseconds

        Returns:
            True if packets received, False if timeout
        """
        start_count = self.packets_received
        start_time = cocotb.utils.get_sim_time('ns')

        while True:
            if self.packets_received >= start_count + count:
                return True

            current_time = cocotb.utils.get_sim_time('ns')
            if current_time - start_time > timeout_ns:
                return False

            await Timer(100, units='ns')

    async def reset(self):
        """Reset receiver state"""
        self.clear_buffers()
        self.reset_statistics()
        self.last_packet_time = 0
        self.frame_start_times.clear()
        self.line_start_times.clear()

        self.logger.info("Receiver reset complete")


class Csi2ImageReceiver(Csi2RxModel):
    """
    Specialized CSI-2 receiver for image data

    Provides higher-level methods for image frame validation and analysis
    """

    def __init__(self, bus: Union[Csi2Bus, Csi2DPhyBus, Csi2CPhyBus],
                 config: Csi2Config):
        super().__init__(bus, config)

        # Image-specific statistics
        self.frame_sizes = defaultdict(list)  # Track frame sizes per VC
        self.frame_rates = defaultdict(list)  # Track frame intervals per VC

    async def validate_frame_format(self, virtual_channel: int,
                                  expected_width: int, expected_height: int,
                                  expected_data_type: DataType) -> bool:
        """
        Validate received frame format

        Args:
            virtual_channel: Virtual channel to check
            expected_width: Expected frame width
            expected_height: Expected frame height
            expected_data_type: Expected data type

        Returns:
            True if format matches expectations
        """
        frame_buffer = self.frame_buffers[virtual_channel]

        if not frame_buffer.is_frame_complete():
            return False

        # Check data type
        if frame_buffer.data_type != expected_data_type:
            self.logger.error(f"Data type mismatch: expected {expected_data_type}, "
                            f"got {frame_buffer.data_type}")
            return False

        # Calculate expected frame size
        bytes_per_pixel = self._get_bytes_per_pixel(expected_data_type)
        expected_size = int(expected_width * expected_height * bytes_per_pixel)
        actual_size = len(frame_buffer.frame_data)

        if actual_size != expected_size:
            self.logger.error(f"Frame size mismatch: expected {expected_size} bytes, "
                            f"got {actual_size} bytes")
            return False

        return True

    def _get_bytes_per_pixel(self, data_type: DataType) -> float:
        """Get bytes per pixel for data type"""
        bytes_per_pixel = {
            DataType.RAW8: 1.0,
            DataType.RAW10: 1.25,
            DataType.RAW12: 1.5,
            DataType.RAW16: 2.0,
            DataType.RGB888: 3.0,
            DataType.RGB565: 2.0,
            DataType.YUV422_8BIT: 2.0,
            DataType.YUV420_8BIT: 1.5,
        }
        return bytes_per_pixel.get(data_type, 1.0)

    def calculate_frame_rate(self, virtual_channel: int) -> Optional[float]:
        """
        Calculate frame rate for virtual channel

        Args:
            virtual_channel: Virtual channel ID

        Returns:
            Frame rate in FPS or None if insufficient data
        """
        if virtual_channel not in self.frame_start_times:
            return None

        timestamps = self.frame_rates[virtual_channel]
        if len(timestamps) < 2:
            return None

        # Calculate average interval
        intervals = [timestamps[i+1] - timestamps[i]
                    for i in range(len(timestamps)-1)]
        avg_interval_ns = sum(intervals) / len(intervals)

        # Convert to FPS
        fps = 1e9 / avg_interval_ns
        return fps

    async def analyze_frame_pattern(self, virtual_channel: int) -> dict:
        """
        Analyze received frame for test patterns

        Args:
            virtual_channel: Virtual channel to analyze

        Returns:
            Pattern analysis results
        """
        frame_data = self.get_frame_data(virtual_channel)
        if not frame_data:
            return {'error': 'No frame data available'}

        # Basic statistics
        pixel_values = list(frame_data)

        analysis = {
            'size_bytes': len(frame_data),
            'min_value': min(pixel_values),
            'max_value': max(pixel_values),
            'avg_value': sum(pixel_values) / len(pixel_values),
            'pattern_type': 'unknown'
        }

        # Pattern detection
        if self._is_ramp_pattern(pixel_values):
            analysis['pattern_type'] = 'ramp'
        elif self._is_checkerboard_pattern(pixel_values):
            analysis['pattern_type'] = 'checkerboard'
        elif len(set(pixel_values)) == 1:
            analysis['pattern_type'] = 'solid'

        return analysis

    def _is_ramp_pattern(self, pixels: List[int]) -> bool:
        """Check if pixels form a ramp pattern"""
        if len(pixels) < 10:
            return False

        # Check if values are monotonically increasing
        increasing = sum(1 for i in range(len(pixels)-1)
                        if pixels[i+1] >= pixels[i])
        return increasing > len(pixels) * 0.8

    def _is_checkerboard_pattern(self, pixels: List[int]) -> bool:
        """Check if pixels form a checkerboard pattern"""
        if len(pixels) < 64:  # Minimum for 8x8 pattern
            return False

        unique_values = set(pixels)
        return len(unique_values) <= 2  # Should have at most 2 colors
