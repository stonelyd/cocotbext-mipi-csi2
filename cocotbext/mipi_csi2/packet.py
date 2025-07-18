"""
MIPI CSI-2 Packet definitions and utilities

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
import struct
from .config import DataType, VirtualChannel
from .utils import calculate_ecc, calculate_checksum, validate_ecc
from .exceptions import Csi2ProtocolError, Csi2EccError


@dataclass
class Csi2PacketHeader:
    """CSI-2 packet header (4 bytes for both short and long packets)"""
    data_id: int        # Data Identifier (DI) - 8 bits
    word_count: int     # Word Count (WC) - 16 bits
    ecc: int           # Error Correction Code - 8 bits

    @property
    def virtual_channel(self) -> int:
        """Extract Virtual Channel from Data Identifier"""
        return (self.data_id >> 6) & 0x03

    @property
    def data_type(self) -> int:
        """Extract Data Type from Data Identifier"""
        return self.data_id & 0x3F

    @classmethod
    def from_vc_dt(cls, virtual_channel: int, data_type: int, word_count: int = 0) -> "Csi2PacketHeader":
        """Create header from Virtual Channel and Data Type"""
        if not 0 <= virtual_channel <= 3:
            raise ValueError(f"Virtual Channel must be 0-3, got {virtual_channel}")
        if not 0 <= data_type <= 0x3F:
            raise ValueError(f"Data Type must be 0-0x3F, got {data_type}")
        if not 0 <= word_count <= 0xFFFF:
            raise ValueError(f"Word Count must be 0-0xFFFF, got {word_count}")

        data_id = (virtual_channel << 6) | data_type
        ecc = calculate_ecc(data_id, word_count)

        return cls(data_id=data_id, word_count=word_count, ecc=ecc)

    def to_bytes(self) -> bytes:
        """Convert header to byte array"""
        return struct.pack('<BHB', self.data_id, self.word_count, self.ecc)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Csi2PacketHeader":
        """Create header from byte array"""
        if len(data) != 4:
            raise ValueError(f"Header must be 4 bytes, got {len(data)}")

        data_id, word_count, ecc = struct.unpack('<BHB', data)
        return cls(data_id=data_id, word_count=word_count, ecc=ecc)

    def validate_ecc(self) -> bool:
        """Validate ECC against header data"""
        return validate_ecc(self.data_id, self.word_count, self.ecc)

    def correct_ecc(self) -> Tuple[bool, Optional["Csi2PacketHeader"]]:
        """Attempt to correct single-bit ECC errors"""
        # For now, we do not support ECC correction. Always return False.
        return False, None


class Csi2Packet(ABC):
    """Abstract base class for CSI-2 packets"""

    def __init__(self, header: Csi2PacketHeader):
        self.header = header
        self.timestamp = None  # Can be set for timing analysis

    @abstractmethod
    def to_bytes(self) -> bytes:
        """Convert packet to byte array"""
        pass

    @abstractmethod
    def get_packet_length(self) -> int:
        """Get total packet length in bytes"""
        pass

    @property
    def virtual_channel(self) -> int:
        """Get virtual channel"""
        return self.header.virtual_channel

    @property
    def data_type(self) -> int:
        """Get data type"""
        return self.header.data_type

    def is_short_packet(self) -> bool:
        """Check if this is a short packet"""
        return isinstance(self, Csi2ShortPacket)

    def is_long_packet(self) -> bool:
        """Check if this is a long packet"""
        return isinstance(self, Csi2LongPacket)


class Csi2ShortPacket(Csi2Packet):
    """CSI-2 Short Packet (4 bytes total)"""

    def __init__(self, virtual_channel: int, data_type: DataType, data: int = 0):
        """
        Create CSI-2 Short Packet

        Args:
            virtual_channel: Virtual channel (0-3)
            data_type: Data type
            data: 16-bit data payload (encoded in word_count field)
        """
        if not 0 <= data <= 0xFFFF:
            raise ValueError(f"Short packet data must be 0-0xFFFF, got {data}")

        header = Csi2PacketHeader.from_vc_dt(virtual_channel, data_type.value, data)
        super().__init__(header)
        self.data = data

    def to_bytes(self) -> bytes:
        """Convert to byte array"""
        return self.header.to_bytes()

    def get_packet_length(self) -> int:
        """Get packet length (always 4 bytes for short packets)"""
        return 4

    @classmethod
    def frame_start(cls, virtual_channel: int = 0, frame_number: int = 0) -> "Csi2ShortPacket":
        """Create Frame Start packet"""
        return cls(virtual_channel, DataType.FRAME_START, frame_number)

    @classmethod
    def frame_end(cls, virtual_channel: int = 0, frame_number: int = 0) -> "Csi2ShortPacket":
        """Create Frame End packet"""
        return cls(virtual_channel, DataType.FRAME_END, frame_number)

    @classmethod
    def line_start(cls, virtual_channel: int = 0, line_number: int = 0) -> "Csi2ShortPacket":
        """Create Line Start packet"""
        return cls(virtual_channel, DataType.LINE_START, line_number)

    @classmethod
    def line_end(cls, virtual_channel: int = 0, line_number: int = 0) -> "Csi2ShortPacket":
        """Create Line End packet"""
        return cls(virtual_channel, DataType.LINE_END, line_number)


class Csi2LongPacket(Csi2Packet):
    """CSI-2 Long Packet (variable length)"""

    def __init__(self, virtual_channel: int, data_type: DataType, payload: bytes, checksum: Optional[int] = None):
        """
        Create CSI-2 Long Packet

        Args:
            virtual_channel: Virtual channel (0-3)
            data_type: Data type
            payload: Packet payload data
            checksum: Optional pre-calculated checksum
        """
        if len(payload) > 65535:
            raise ValueError(f"Payload too large: {len(payload)} bytes")

        header = Csi2PacketHeader.from_vc_dt(virtual_channel, data_type.value, len(payload))
        super().__init__(header)
        self.payload = payload
        if checksum is None:
            self.checksum = calculate_checksum(payload)
        else:
            self.checksum = checksum

    def to_bytes(self) -> bytes:
        """Convert to byte array"""
        return self.header.to_bytes() + self.payload + struct.pack('<H', self.checksum)

    def get_packet_length(self) -> int:
        """Get total packet length"""
        return 4 + len(self.payload) + 2  # Header + payload + checksum

    def validate_checksum(self) -> bool:
        """Validate packet checksum"""
        calculated = calculate_checksum(self.payload)
        return self.checksum == calculated

    @classmethod
    def create_image_data(cls, virtual_channel: int, data_type: DataType,
                         width: int, height: int, pixel_data: bytes) -> "Csi2LongPacket":
        """Create image data packet"""
        if len(pixel_data) != width * height * cls._get_bytes_per_pixel(data_type):
            raise ValueError("Pixel data size doesn't match width/height/format")

        return cls(virtual_channel, data_type, pixel_data)

    @staticmethod
    def _get_bytes_per_pixel(data_type: DataType) -> float:
        """Get bytes per pixel for different formats"""
        bytes_per_pixel = {
            DataType.RAW8: 1.0,
            DataType.RAW10: 1.25,  # Packed format
            DataType.RAW12: 1.5,   # Packed format
            DataType.RAW16: 2.0,
            DataType.RGB888: 3.0,
            DataType.RGB565: 2.0,
            DataType.YUV422_8BIT: 2.0,
            DataType.YUV420_8BIT: 1.5,
        }
        return bytes_per_pixel.get(data_type, 1.0)


class Csi2PacketBuilder:
    """Builder class for creating CSI-2 packets"""

    def __init__(self):
        self.virtual_channel = 0
        self.inject_ecc_error = False
        self.inject_checksum_error = False

    def set_virtual_channel(self, vc: int) -> "Csi2PacketBuilder":
        """Set virtual channel for subsequent packets"""
        self.virtual_channel = vc
        return self

    def enable_ecc_error_injection(self, enable: bool = True) -> "Csi2PacketBuilder":
        """Enable ECC error injection"""
        self.inject_ecc_error = enable
        return self

    def enable_checksum_error_injection(self, enable: bool = True) -> "Csi2PacketBuilder":
        """Enable checksum error injection"""
        self.inject_checksum_error = enable
        return self

    def build_frame_start(self, frame_number: int = 0) -> Csi2ShortPacket:
        """Build Frame Start packet"""
        packet = Csi2ShortPacket.frame_start(self.virtual_channel, frame_number)
        if self.inject_ecc_error:
            packet.header.ecc ^= 0x01  # Flip one bit
        return packet

    def build_frame_end(self, frame_number: int = 0) -> Csi2ShortPacket:
        """Build Frame End packet"""
        packet = Csi2ShortPacket.frame_end(self.virtual_channel, frame_number)
        if self.inject_ecc_error:
            packet.header.ecc ^= 0x01
        return packet

    def build_line_start(self, line_number: int = 0) -> Csi2ShortPacket:
        """Build Line Start packet"""
        packet = Csi2ShortPacket.line_start(self.virtual_channel, line_number)
        if self.inject_ecc_error:
            packet.header.ecc ^= 0x01
        return packet

    def build_line_end(self, line_number: int = 0) -> Csi2ShortPacket:
        """Build Line End packet"""
        packet = Csi2ShortPacket.line_end(self.virtual_channel, line_number)
        if self.inject_ecc_error:
            packet.header.ecc ^= 0x01
        return packet

    def build_pixel_data(self, data_type: DataType, payload: bytes) -> Csi2LongPacket:
        """Build pixel data packet"""
        packet = Csi2LongPacket(self.virtual_channel, data_type, payload)
        if self.inject_ecc_error:
            packet.header.ecc ^= 0x01
        if self.inject_checksum_error:
            packet.checksum ^= 0x0001
        return packet

    def build_raw8_line(self, width: int, line_data: bytes) -> Csi2LongPacket:
        """Build RAW8 image line packet"""
        if len(line_data) != width:
            raise ValueError(f"Line data length {len(line_data)} doesn't match width {width}")
        return self.build_pixel_data(DataType.RAW8, line_data)

    def build_rgb888_line(self, width: int, line_data: bytes) -> Csi2LongPacket:
        """Build RGB888 image line packet"""
        if len(line_data) != width * 3:
            raise ValueError(f"Line data length {len(line_data)} doesn't match width*3 {width*3}")
        return self.build_pixel_data(DataType.RGB888, line_data)


class Csi2PacketParser:
    """Parser for CSI-2 packet streams"""

    def __init__(self):
        self.packet_buffer = bytearray()
        self.parsed_packets = []
        self.error_count = 0
        self.ecc_error_count = 0
        self.checksum_error_count = 0

    def feed_data(self, data: bytes) -> List[Csi2Packet]:
        """Feed raw data and return parsed packets"""
        self.packet_buffer.extend(data)
        new_packets = []

        while len(self.packet_buffer) >= 4:  # Minimum packet size
            try:
                packet = self._parse_next_packet()
                if packet:
                    new_packets.append(packet)
                    self.parsed_packets.append(packet)
                else:
                    break  # Not enough data for complete packet
            except Exception as e:
                # Skip malformed data
                self.error_count += 1
                # Clear the buffer to prevent loops on corrupted data
                self.packet_buffer.clear()
                break

        return new_packets

    def _parse_next_packet(self) -> Optional[Csi2Packet]:
        """Parse the next packet from buffer"""
        if len(self.packet_buffer) < 4:
            return None

        # Parse header
        try:
            header = Csi2PacketHeader.from_bytes(self.packet_buffer[:4])
        except Exception:
            raise Csi2ProtocolError("Invalid packet header")

        # Validate ECC
        if not header.validate_ecc():
            self.ecc_error_count += 1
            # Try to correct single-bit errors
            corrected, corrected_header = header.correct_ecc()
            if corrected and corrected_header:
                header = corrected_header
            else:
                raise Csi2EccError("Uncorrectable ECC error")

        # Check if short or long packet
        data_type = header.data_type
        if self._is_short_packet_type(data_type):
            # Short packet - just the header
            packet = self._create_short_packet(header)
            del self.packet_buffer[:4]
            return packet
        else:
            # Long packet - need payload + checksum
            packet_size = 4 + header.word_count + 2
            if len(self.packet_buffer) < packet_size:
                return None  # Not enough data yet

            payload = bytes(self.packet_buffer[4:4 + header.word_count])
            checksum_bytes = self.packet_buffer[4 + header.word_count:4 + header.word_count + 2]
            checksum = struct.unpack('<H', checksum_bytes)[0]

            packet = self._create_long_packet(header, payload, checksum)
            del self.packet_buffer[:packet_size]
            return packet

    def _is_short_packet_type(self, data_type: int) -> bool:
        """Check if data type indicates short packet"""
        short_packet_types = {
            DataType.FRAME_START.value,
            DataType.FRAME_END.value,
            DataType.LINE_START.value,
            DataType.LINE_END.value,
        }
        return data_type in short_packet_types or (0x08 <= data_type <= 0x0F)

    def _create_short_packet(self, header: Csi2PacketHeader) -> Csi2ShortPacket:
        """Create short packet from header"""
        # The data is in the word_count field for short packets
        return Csi2ShortPacket(
            header.virtual_channel,
            DataType(header.data_type),
            header.word_count
        )

    def _create_long_packet(self, header: Csi2PacketHeader,
                          payload: bytes, checksum: int) -> Csi2LongPacket:
        """Create long packet from components"""
        packet = Csi2LongPacket(
            header.virtual_channel,
            DataType(header.data_type),
            payload
        )
        packet.checksum = checksum

        # Validate checksum
        if not packet.validate_checksum():
            self.checksum_error_count += 1
            # Could raise exception or just log error

        return packet

    def get_statistics(self) -> dict:
        """Get parsing statistics"""
        return {
            'total_packets': len(self.parsed_packets),
            'error_count': self.error_count,
            'ecc_error_count': self.ecc_error_count,
            'checksum_error_count': self.checksum_error_count,
            'buffer_size': len(self.packet_buffer)
        }

    def reset(self):
        """Reset parser state"""
        self.packet_buffer.clear()
        self.parsed_packets.clear()
        self.error_count = 0
        self.ecc_error_count = 0
        self.checksum_error_count = 0
