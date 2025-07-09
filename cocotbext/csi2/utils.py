"""
MIPI CSI-2 Utility functions

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

import numpy as np
from typing import List, Tuple, Union
import struct
import logging


def calculate_ecc(data_id: int, word_count: int) -> int:
    """
    Calculate 8-bit Error Correction Code (ECC) for CSI-2 packet header.

    This implementation is based on the MIPI CSI-2 Specification.

    Args:
        data_id: 8-bit Data Identifier (DI).
        word_count: 16-bit Word Count (WC).

    Returns:
        8-bit ECC value.
    """
    # The 24 bits of data for which ECC is calculated
    # D[23:16] = Data Identifier (DI[7:0])
    # D[15:0] = Word Count (WC[15:0])
    data = (data_id << 16) | word_count

    # Extracting each bit into a list, d[0] is the LSB.
    d = [(data >> i) & 1 for i in range(24)]

    # Parity bits based on the MIPI CSI-2 specification's generator matrix
    p = [0] * 6
    p[0] = d[10] ^ d[8] ^ d[7] ^ d[6] ^ d[4] ^ d[2] ^ d[1] ^ d[0] ^ d[22] ^ d[21] ^ d[20] ^ d[18] ^ d[17] ^ d[14] ^ d[13] ^ d[12]
    p[1] = d[11] ^ d[9] ^ d[7] ^ d[6] ^ d[5] ^ d[3] ^ d[2] ^ d[1] ^ d[23] ^ d[21] ^ d[20] ^ d[19] ^ d[17] ^ d[15] ^ d[14] ^ d[13]
    p[2] = d[11] ^ d[10] ^ d[9] ^ d[8] ^ d[7] ^ d[5] ^ d[4] ^ d[3] ^ d[23] ^ d[22] ^ d[20] ^ d[19] ^ d[18] ^ d[16] ^ d[15] ^ d[14]
    p[3] = d[15] ^ d[14] ^ d[13] ^ d[12] ^ d[11] ^ d[10] ^ d[9] ^ d[8] ^ d[7] ^ d[6] ^ d[5] ^ d[4] ^ d[23] ^ d[22] ^ d[21] ^ d[20]
    p[4] = d[19] ^ d[18] ^ d[17] ^ d[16] ^ d[15] ^ d[13] ^ d[12] ^ d[11] ^ d[6] ^ d[3] ^ d[1] ^ d[0] ^ d[23] ^ d[22] ^ d[21]
    p[5] = d[20] ^ d[19] ^ d[18] ^ d[17] ^ d[16] ^ d[15] ^ d[14] ^ d[10] ^ d[9] ^ d[8] ^ d[5] ^ d[2] ^ d[0] ^ d[23] ^ d[22]

    return (p[5] << 5) | (p[4] << 4) | (p[3] << 3) | (p[2] << 2) | (p[1] << 1) | p[0]


def validate_ecc(data_id: int, word_count: int, received_ecc: int) -> bool:
    """
    Validate ECC against received data

    Args:
        data_id: 8-bit Data Identifier
        word_count: 16-bit Word Count
        received_ecc: 8-bit received ECC

    Returns:
        True if ECC is valid
    """
    calculated_ecc = calculate_ecc(data_id, word_count)
    return calculated_ecc == received_ecc


def calculate_checksum(data: bytes) -> int:
    """
    Calculate 16-bit checksum for CSI-2 long packet payload

    Uses simple sum algorithm as specified in CSI-2

    Args:
        data: Payload data bytes

    Returns:
        16-bit checksum value
    """
    checksum = 0
    for byte in data:
        checksum += byte

    return checksum & 0xFFFF


def validate_checksum(data: bytes, received_checksum: int) -> bool:
    """
    Validate checksum against received data

    Args:
        data: Payload data bytes
        received_checksum: 16-bit received checksum

    Returns:
        True if checksum is valid
    """
    calculated_checksum = calculate_checksum(data)
    return calculated_checksum == received_checksum


def bytes_to_lanes(data: bytes, lane_count: int) -> List[bytes]:
    """
    Distribute byte data across multiple lanes

    Args:
        data: Input data bytes
        lane_count: Number of lanes (1, 2, 4, 8)

    Returns:
        List of byte arrays, one per lane
    """
    if lane_count not in [1, 2, 4, 8]:
        raise ValueError(f"Invalid lane count: {lane_count}")

    if lane_count == 1:
        return [data]

    # Distribute bytes round-robin across lanes
    lanes = [bytearray() for _ in range(lane_count)]

    for i, byte in enumerate(data):
        lane_idx = i % lane_count
        lanes[lane_idx].append(byte)

    return [bytes(lane) for lane in lanes]


def lanes_to_bytes(lane_data: List[bytes]) -> bytes:
    """
    Merge data from multiple lanes back into single byte stream

    Args:
        lane_data: List of byte arrays from each lane

    Returns:
        Merged byte data
    """
    if not lane_data:
        return b''

    if len(lane_data) == 1:
        return lane_data[0]

    # Merge bytes round-robin from lanes
    result = bytearray()
    max_len = max(len(lane) for lane in lane_data)

    for i in range(max_len):
        for lane in lane_data:
            if i < len(lane):
                result.append(lane[i])

    return bytes(result)


def pack_raw10(pixels: List[int]) -> bytes:
    """
    Pack 10-bit RAW pixels into CSI-2 RAW10 format

    RAW10 packs 4 pixels into 5 bytes:
    - Byte 0: P0[9:2]
    - Byte 1: P1[9:2]
    - Byte 2: P2[9:2]
    - Byte 3: P3[9:2]
    - Byte 4: P3[1:0]P2[1:0]P1[1:0]P0[1:0]

    Args:
        pixels: List of 10-bit pixel values

    Returns:
        Packed byte data
    """
    if len(pixels) % 4 != 0:
        # Pad with zeros to multiple of 4
        pixels = pixels + [0] * (4 - len(pixels) % 4)

    packed = bytearray()

    for i in range(0, len(pixels), 4):
        p0, p1, p2, p3 = pixels[i:i+4]

        # Pack high 8 bits of each pixel
        packed.append((p0 >> 2) & 0xFF)
        packed.append((p1 >> 2) & 0xFF)
        packed.append((p2 >> 2) & 0xFF)
        packed.append((p3 >> 2) & 0xFF)

        # Pack low 2 bits
        low_bits = ((p3 & 3) << 6) | ((p2 & 3) << 4) | ((p1 & 3) << 2) | (p0 & 3)
        packed.append(low_bits)

    return bytes(packed)


def unpack_raw10(data: bytes) -> List[int]:
    """
    Unpack CSI-2 RAW10 format into 10-bit pixel values

    Args:
        data: Packed RAW10 byte data

    Returns:
        List of 10-bit pixel values
    """
    if len(data) % 5 != 0:
        raise ValueError("RAW10 data length must be multiple of 5")

    pixels = []

    for i in range(0, len(data), 5):
        bytes_group = data[i:i+5]

        # Extract high 8 bits
        p0_high = bytes_group[0]
        p1_high = bytes_group[1]
        p2_high = bytes_group[2]
        p3_high = bytes_group[3]

        # Extract low 2 bits
        low_bits = bytes_group[4]
        p0_low = low_bits & 3
        p1_low = (low_bits >> 2) & 3
        p2_low = (low_bits >> 4) & 3
        p3_low = (low_bits >> 6) & 3

        # Combine high and low bits
        pixels.extend([
            (p0_high << 2) | p0_low,
            (p1_high << 2) | p1_low,
            (p2_high << 2) | p2_low,
            (p3_high << 2) | p3_low
        ])

    return pixels


def pack_raw12(pixels: List[int]) -> bytes:
    """
    Pack 12-bit RAW pixels into CSI-2 RAW12 format

    RAW12 packs 2 pixels into 3 bytes:
    - Byte 0: P0[11:4]
    - Byte 1: P1[11:4]
    - Byte 2: P1[3:0]P0[3:0]

    Args:
        pixels: List of 12-bit pixel values

    Returns:
        Packed byte data
    """
    if len(pixels) % 2 != 0:
        pixels = pixels + [0]  # Pad with zero

    packed = bytearray()

    for i in range(0, len(pixels), 2):
        p0, p1 = pixels[i:i+2]

        # Pack high 8 bits
        packed.append((p0 >> 4) & 0xFF)
        packed.append((p1 >> 4) & 0xFF)

        # Pack low 4 bits
        low_bits = ((p1 & 0xF) << 4) | (p0 & 0xF)
        packed.append(low_bits)

    return bytes(packed)


def unpack_raw12(data: bytes) -> List[int]:
    """
    Unpack CSI-2 RAW12 format into 12-bit pixel values

    Args:
        data: Packed RAW12 byte data

    Returns:
        List of 12-bit pixel values
    """
    if len(data) % 3 != 0:
        raise ValueError("RAW12 data length must be multiple of 3")

    pixels = []

    for i in range(0, len(data), 3):
        bytes_group = data[i:i+3]

        # Extract high 8 bits
        p0_high = bytes_group[0]
        p1_high = bytes_group[1]

        # Extract low 4 bits
        low_bits = bytes_group[2]
        p0_low = low_bits & 0xF
        p1_low = (low_bits >> 4) & 0xF

        # Combine high and low bits
        pixels.extend([
            (p0_high << 4) | p0_low,
            (p1_high << 4) | p1_low
        ])

    return pixels


def generate_test_pattern(width: int, height: int, pattern_type: str = "ramp") -> bytes:
    """
    Generate test pattern data for CSI-2 testing

    Args:
        width: Image width in pixels
        height: Image height in pixels
        pattern_type: Type of pattern ("ramp", "checkerboard", "solid")

    Returns:
        Test pattern data as bytes (RAW8 format)
    """
    data = bytearray()

    if pattern_type == "ramp":
        for y in range(height):
            for x in range(width):
                # Horizontal ramp
                value = (x * 255) // width
                data.append(value)

    elif pattern_type == "checkerboard":
        for y in range(height):
            for x in range(width):
                # 8x8 checkerboard
                if ((x // 8) + (y // 8)) % 2:
                    data.append(255)
                else:
                    data.append(0)

    elif pattern_type == "solid":
        # Solid gray
        for _ in range(width * height):
            data.append(128)

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return bytes(data)


def dphy_encode_byte(byte_val: int) -> Tuple[int, int]:
    """
    Encode byte for D-PHY transmission (simplified)

    Args:
        byte_val: 8-bit value to encode

    Returns:
        Tuple of (positive_edge_data, negative_edge_data)
    """
    # Simplified encoding - actual D-PHY encoding is more complex
    # This is just for simulation purposes
    return byte_val, ~byte_val & 0xFF


def cphy_encode_symbol(symbol: int) -> Tuple[int, int, int]:
    """
    Encode symbol for C-PHY transmission (simplified)

    Args:
        symbol: Symbol value to encode

    Returns:
        Tuple of (trio_a, trio_b, trio_c) values
    """
    # Simplified C-PHY encoding
    # Actual C-PHY uses complex state machine encoding
    a = symbol & 0x03
    b = (symbol >> 2) & 0x03
    c = (symbol >> 4) & 0x03
    return a, b, c


def create_image_frame_sequence(width: int, height: int,
                              virtual_channel: int = 0,
                              data_type = None) -> List:
    """
    Create a complete image frame packet sequence

    Args:
        width: Image width
        height: Image height
        virtual_channel: Virtual channel ID
        data_type: Pixel data type

    Returns:
        List of CSI-2 packets for complete frame
    """
    from .packet import Csi2PacketBuilder, DataType

    if data_type is None:
        data_type = DataType.RAW8

    builder = Csi2PacketBuilder().set_virtual_channel(virtual_channel)
    packets = []

    # Frame start
    packets.append(builder.build_frame_start(0))

    # Line data
    for line in range(height):
        # Line start
        packets.append(builder.build_line_start(line))

        # Generate line data
        line_data = generate_test_pattern(width, 1)
        packets.append(builder.build_pixel_data(data_type, line_data))

        # Line end
        packets.append(builder.build_line_end(line))

    # Frame end
    packets.append(builder.build_frame_end(0))

    return packets


def setup_logging(level: int = logging.INFO):
    """Setup logging for CSI-2 extension"""
    logger = logging.getLogger('cocotbext.csi2')
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
