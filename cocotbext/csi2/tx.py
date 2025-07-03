"""
MIPI CSI-2 Transmitter Model

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
from cocotb.triggers import Timer, Event
from cocotb.queue import Queue
import asyncio
import random
import logging
from typing import List, Optional, Union, Callable
from .bus import Csi2Bus, Csi2DPhyBus, Csi2CPhyBus
from .config import Csi2Config, PhyType, DataType, VirtualChannel
from .packet import Csi2Packet, Csi2ShortPacket, Csi2LongPacket, Csi2PacketBuilder
from .phy import DPhyModel, CPhyModel
from .exceptions import Csi2Exception, Csi2ProtocolError, Csi2PhyError
from .utils import create_image_frame_sequence, generate_test_pattern


class Csi2TxModel:
    """
    CSI-2 Transmitter Model for testing receiver implementations

    This model can generate CSI-2 packet streams with configurable timing,
    error injection, and multi-lane distribution.
    """

    def __init__(self, bus: Union[Csi2Bus, Csi2DPhyBus, Csi2CPhyBus],
                 config: Csi2Config, phy_model=None):
        """
        Initialize CSI-2 transmitter model

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
            if isinstance(phy_model, DPhyModel):
                self.phy_bus = phy_model.bus
            elif isinstance(phy_model, CPhyModel):
                self.phy_bus = phy_model.bus
            else:
                raise Csi2ProtocolError(f"Unsupported PHY model type: {type(phy_model)}")
        else:
            # Initialize PHY model based on configuration
            if config.phy_type == PhyType.DPHY:
                if not isinstance(bus, Csi2DPhyBus):
                    # Create D-PHY bus if generic bus provided
                    self.phy_bus = Csi2DPhyBus(bus.entity, config.lane_count)
                else:
                    self.phy_bus = bus
                self.phy_model = DPhyModel(self.phy_bus, config)
            elif config.phy_type == PhyType.CPHY:
                if not isinstance(bus, Csi2CPhyBus):
                    # Create C-PHY bus if generic bus provided
                    self.phy_bus = Csi2CPhyBus(bus.entity, config.trio_count)
                else:
                    self.phy_bus = bus
                self.phy_model = CPhyModel(self.phy_bus, config)
            else:
                raise Csi2ProtocolError(f"Unsupported PHY type: {config.phy_type}")

        # Packet queues for different virtual channels
        self.tx_queues = {vc: Queue() for vc in range(16)}

        # Statistics and monitoring
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors_injected = 0
        self.frame_count = 0

        # Control flags
        self.transmitting = False
        self.continuous_mode = False

        # Events
        self.frame_start_event = Event()
        self.frame_end_event = Event()
        self.transmission_complete = Event()

        # Callbacks
        self.on_packet_sent: Optional[Callable] = None
        self.on_frame_start: Optional[Callable] = None
        self.on_frame_end: Optional[Callable] = None
        self.on_error_injected: Optional[Callable] = None

        # Packet builder for convenience
        self.packet_builder = Csi2PacketBuilder()

        self.logger = logging.getLogger('cocotbext.csi2.tx')

        # Start transmission task
        self._tx_task = cocotb.start_soon(self._transmission_handler())

    async def send_packet(self, packet: Csi2Packet, virtual_channel: Optional[int] = None):
        """
        Queue a packet for transmission

        Args:
            packet: CSI-2 packet to send
            virtual_channel: Override virtual channel (uses packet's VC if None)
        """
        vc = virtual_channel if virtual_channel is not None else packet.virtual_channel
        if vc not in self.tx_queues:
            raise Csi2ProtocolError(f"Invalid virtual channel: {vc}")

        self.logger.info(f"TX: Queuing packet: VC{vc} DT{packet.data_type:02X}")
        await self.tx_queues[vc].put(packet)
        self.logger.info(f"TX: Packet queued successfully")

    async def send_packets(self, packets: List[Csi2Packet], virtual_channel: Optional[int] = None):
        """
        Queue multiple packets for transmission

        Args:
            packets: List of CSI-2 packets to send
            virtual_channel: Override virtual channel for all packets
        """
        for packet in packets:
            await self.send_packet(packet, virtual_channel)

    async def send_frame(self, width: int, height: int, data_type: DataType = DataType.RAW8,
                        virtual_channel: int = 0, frame_number: int = 0,
                        pixel_data: Optional[bytes] = None):
        """
        Send a complete image frame

        Args:
            width: Image width in pixels
            height: Image height in lines
            data_type: Pixel data type
            virtual_channel: Virtual channel ID
            frame_number: Frame sequence number
            pixel_data: Optional pixel data (generates test pattern if None)
        """
        self.logger.info(f"Sending frame {frame_number}: {width}x{height} VC{virtual_channel}")

        builder = self.packet_builder.set_virtual_channel(virtual_channel)

        # Frame start
        await self.send_packet(builder.build_frame_start(frame_number))

        if self.on_frame_start:
            await self.on_frame_start(frame_number, virtual_channel)

        # Send image lines
        bytes_per_pixel = self._get_bytes_per_pixel(data_type)
        line_bytes = int(width * bytes_per_pixel)

        for line_num in range(height):
            # Line start
            await self.send_packet(builder.build_line_start(line_num + 1))

            # Generate or use provided pixel data
            if pixel_data:
                line_data = pixel_data[line_num * line_bytes:(line_num + 1) * line_bytes]
            else:
                line_data = generate_test_pattern(width, 1, "ramp")

            # Line data packet
            await self.send_packet(builder.build_pixel_data(data_type, line_data))

            # Line end
            await self.send_packet(builder.build_line_end(line_num + 1))

        # Frame end
        await self.send_packet(builder.build_frame_end(frame_number))

        if self.on_frame_end:
            await self.on_frame_end(frame_number, virtual_channel)

        self.frame_count += 1

    async def send_test_sequence(self, pattern_type: str = "basic"):
        """
        Send predefined test sequences

        Args:
            pattern_type: Type of test pattern ("basic", "multi_vc", "error_test")
        """
        if pattern_type == "basic":
            await self._send_basic_test_sequence()
        elif pattern_type == "multi_vc":
            await self._send_multi_vc_test_sequence()
        elif pattern_type == "error_test":
            await self._send_error_test_sequence()
        else:
            raise ValueError(f"Unknown test pattern: {pattern_type}")

    async def _send_basic_test_sequence(self):
        """Send basic test sequence with single frame"""
        await self.send_frame(640, 480, DataType.RAW8, 0, 0)

    async def _send_multi_vc_test_sequence(self):
        """Send test sequence with multiple virtual channels"""
        # Send frames on different VCs
        tasks = []
        for vc in range(4):
            task = cocotb.start_soon(
                self.send_frame(320, 240, DataType.RAW8, vc, vc)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _send_error_test_sequence(self):
        """Send test sequence with intentional errors"""
        # Enable error injection temporarily
        original_rate = self.config.error_injection_rate
        self.config.error_injection_rate = 0.1

        try:
            await self.send_frame(160, 120, DataType.RAW8, 0, 0)
        finally:
            self.config.error_injection_rate = original_rate

    def start_continuous_transmission(self, frame_rate_fps: float = 30.0,
                                    width: int = 640, height: int = 480,
                                    data_type: DataType = DataType.RAW8):
        """
        Start continuous frame transmission

        Args:
            frame_rate_fps: Target frame rate
            width: Frame width
            height: Frame height
            data_type: Pixel data type
        """
        self.continuous_mode = True
        self.continuous_config = {
            'frame_rate_fps': frame_rate_fps,
            'width': width,
            'height': height,
            'data_type': data_type
        }

        # Start continuous transmission task
        cocotb.start_soon(self._continuous_transmission_handler())

    def stop_continuous_transmission(self):
        """Stop continuous frame transmission"""
        self.continuous_mode = False

    async def _continuous_transmission_handler(self):
        """Handle continuous frame transmission"""
        frame_period_ns = 1e9 / self.continuous_config['frame_rate_fps']
        frame_num = 0

        while self.continuous_mode:
            await self.send_frame(
                self.continuous_config['width'],
                self.continuous_config['height'],
                self.continuous_config['data_type'],
                0,
                frame_num
            )

            frame_num += 1
            await Timer(frame_period_ns, units='ns')

    async def _transmission_handler(self):
        """Main transmission handler task"""
        self.logger.info("TX: Transmission handler started")
        while True:
            # Check all virtual channels for pending packets
            packet_to_send = None
            vc_to_send = -1

            for vc in range(16):
                if not self.tx_queues[vc].empty():
                    packet_to_send = await self.tx_queues[vc].get()
                    vc_to_send = vc
                    self.logger.info(f"TX: Found packet in queue VC{vc}")
                    break  # Process one packet at a time

            if packet_to_send:
                self.logger.info(f"TX: Transmitting 1 packet")
                await self._transmit_packets([(vc_to_send, packet_to_send)])
            else:
                # No packets to send, wait a bit
                await Timer(100, units='ns')

    async def _transmit_packets(self, packets: List[tuple]):
        """
        Transmit packets over PHY layer

        Args:
            packets: List of (virtual_channel, packet) tuples
        """
        self.transmitting = True

        try:
            # Start PHY transmission
            await self.phy_model.start_packet_transmission()

            # Send each packet
            for vc, packet in packets:
                await self._transmit_single_packet(packet)

                # Apply inter-packet spacing
                if len(packets) > 1:
                    spacing_ns = self.config.clock_period_ns * 10
                    await Timer(spacing_ns, units='ns')

            # Stop PHY transmission
            await self.phy_model.stop_packet_transmission()

        except Exception as e:
            self.logger.error(f"Transmission error: {e}")
            raise Csi2PhyError(f"Failed to transmit packets: {e}")
        finally:
            self.transmitting = False
            self.transmission_complete.set()

    async def _transmit_single_packet(self, packet: Csi2Packet):
        """
        Transmit a single packet

        Args:
            packet: Packet to transmit
        """
        # Apply error injection if configured
        if self.config.error_injection_rate > 0:
            if random.random() < self.config.error_injection_rate:
                packet = self._inject_packet_error(packet)
                self.errors_injected += 1
                if self.on_error_injected:
                    await self.on_error_injected(packet)

        # Convert packet to bytes
        packet_bytes = packet.to_bytes()

        # Send via PHY layer
        self.logger.info(f"TX: Sending packet via PHY: VC{packet.virtual_channel} DT{packet.data_type:02X} {len(packet_bytes)} bytes")
        await self.phy_model.send_packet_data(packet_bytes)
        self.logger.info(f"TX: Packet sent via PHY successfully")

        # Debug: print packet bytes
        self.logger.info(f"TX: Packet bytes: {[f'0x{b:02x}' for b in packet_bytes]}")

        # Update statistics
        self.packets_sent += 1
        self.bytes_sent += len(packet_bytes)

        # Trigger callback
        if self.on_packet_sent:
            await self.on_packet_sent(packet)

        self.logger.debug(f"Sent packet: VC{packet.virtual_channel} "
                         f"DT{packet.data_type:02X} {len(packet_bytes)} bytes")

    def _inject_packet_error(self, packet: Csi2Packet) -> Csi2Packet:
        """
        Inject errors into packet for testing

        Args:
            packet: Original packet

        Returns:
            Packet with injected errors
        """
        error_type = random.choice(['ecc', 'checksum', 'data'])

        if error_type == 'ecc':
            # Corrupt ECC
            packet.header.ecc ^= 0x01
        elif error_type == 'checksum' and isinstance(packet, Csi2LongPacket):
            # Corrupt checksum
            packet.checksum ^= 0x0001
        elif error_type == 'data' and isinstance(packet, Csi2LongPacket):
            # Corrupt payload data
            if packet.payload:
                payload = bytearray(packet.payload)
                error_pos = random.randint(0, len(payload) - 1)
                payload[error_pos] ^= 0xFF
                packet.payload = bytes(payload)

        return packet

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

    def enable_error_injection(self, rate: float = 0.01):
        """
        Enable error injection for testing

        Args:
            rate: Error injection rate (0.0 to 1.0)
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Error rate must be 0.0-1.0, got {rate}")

        self.config.error_injection_rate = rate
        self.config.inject_ecc_errors = rate > 0
        self.config.inject_checksum_errors = rate > 0

        self.logger.info(f"Error injection enabled at rate {rate}")

    def disable_error_injection(self):
        """Disable error injection"""
        self.config.error_injection_rate = 0.0
        self.config.inject_ecc_errors = False
        self.config.inject_checksum_errors = False

        self.logger.info("Error injection disabled")

    async def wait_transmission_complete(self):
        """Wait for current transmission to complete"""
        if self.transmitting:
            await self.transmission_complete.wait()
            self.transmission_complete.clear()

    def get_statistics(self) -> dict:
        """Get transmission statistics"""
        return {
            'packets_sent': self.packets_sent,
            'bytes_sent': self.bytes_sent,
            'errors_injected': self.errors_injected,
            'frames_sent': self.frame_count,
            'queue_depths': {vc: self.tx_queues[vc].qsize()
                           for vc in range(16) if not self.tx_queues[vc].empty()},
            'transmitting': self.transmitting,
            'continuous_mode': self.continuous_mode
        }

    def reset_statistics(self):
        """Reset transmission statistics"""
        self.packets_sent = 0
        self.bytes_sent = 0
        self.errors_injected = 0
        self.frame_count = 0

    async def reset(self):
        """Reset transmitter state"""
        self.stop_continuous_transmission()
        await self.wait_transmission_complete()

        # Clear all queues
        for vc in range(16):
            while not self.tx_queues[vc].empty():
                await self.tx_queues[vc].get()

        self.reset_statistics()
        self.logger.info("Transmitter reset complete")


class Csi2ImageTransmitter(Csi2TxModel):
    """
    Specialized CSI-2 transmitter for image data

    Provides higher-level methods for common image transmission scenarios
    """

    def __init__(self, bus: Union[Csi2Bus, Csi2DPhyBus, Csi2CPhyBus],
                 config: Csi2Config):
        super().__init__(bus, config)

        # Image-specific configuration
        self.default_width = 640
        self.default_height = 480
        self.default_data_type = DataType.RAW8
        self.default_frame_rate = 30.0

    async def send_raw8_frame(self, width: int, height: int,
                            pixel_data: Optional[bytes] = None,
                            virtual_channel: int = 0, frame_number: int = 0):
        """Send RAW8 format frame"""
        await self.send_frame(width, height, DataType.RAW8,
                            virtual_channel, frame_number, pixel_data)

    async def send_rgb888_frame(self, width: int, height: int,
                              pixel_data: Optional[bytes] = None,
                              virtual_channel: int = 0, frame_number: int = 0):
        """Send RGB888 format frame"""
        await self.send_frame(width, height, DataType.RGB888,
                            virtual_channel, frame_number, pixel_data)

    async def send_yuv422_frame(self, width: int, height: int,
                              pixel_data: Optional[bytes] = None,
                              virtual_channel: int = 0, frame_number: int = 0):
        """Send YUV422 format frame"""
        await self.send_frame(width, height, DataType.YUV422_8BIT,
                            virtual_channel, frame_number, pixel_data)

    def start_video_stream(self, width: int = None, height: int = None,
                          data_type: DataType = None, frame_rate_fps: float = None):
        """
        Start continuous video stream with default parameters

        Args:
            width: Frame width (uses default if None)
            height: Frame height (uses default if None)
            data_type: Pixel format (uses default if None)
            frame_rate_fps: Frame rate (uses default if None)
        """
        self.start_continuous_transmission(
            frame_rate_fps or self.default_frame_rate,
            width or self.default_width,
            height or self.default_height,
            data_type or self.default_data_type
        )

    async def send_test_patterns(self, patterns: List[str] = None):
        """
        Send various test patterns

        Args:
            patterns: List of pattern names, uses defaults if None
        """
        if patterns is None:
            patterns = ["ramp", "checkerboard", "solid"]

        for i, pattern in enumerate(patterns):
            pixel_data = generate_test_pattern(
                self.default_width, self.default_height, pattern
            )
            await self.send_raw8_frame(
                self.default_width, self.default_height,
                pixel_data, 0, i
            )
