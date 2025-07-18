#!/usr/bin/env python3
"""
Comprehensive CSI-2 Extension Usage Example

This example demonstrates all major features of the cocotbext-csi2 extension
including D-PHY and C-PHY configurations, packet creation, error injection,
and multi-lane operations.
"""

import asyncio
from cocotbext.mipi_csi2 import *
from cocotbext.mipi_csi2.config import Csi2Config, PhyType, DataType, VirtualChannel
from cocotbext.mipi_csi2.packet import Csi2ShortPacket, Csi2LongPacket
from cocotbext.mipi_csi2.utils import calculate_ecc, validate_checksum, generate_test_pattern
from cocotbext.mipi_csi2.exceptions import *


def demonstrate_basic_features():
    """Demonstrate basic CSI-2 functionality"""

    print("=== CSI-2 Extension Feature Demonstration ===\n")

    # 1. Configuration Examples
    print("1. CONFIGURATION EXAMPLES")
    print("-" * 30)

    # D-PHY configuration
    dphy_config = Csi2Config(phy_type=PhyType.DPHY,
                             lane_count=4,
                             bit_rate_mbps=2500,
                             continuous_clock=True,
                             scrambling_enabled=True,
                             virtual_channel_extension=True)
    print(
        f"D-PHY: {dphy_config.lane_count} lanes @ {dphy_config.bit_rate_mbps} Mbps"
    )
    print(f"       Bit period: {dphy_config.get_bit_period_ns():.3f} ns")
    print(f"       Features: Scrambling={dphy_config.scrambling_enabled}, "
          f"Extended VC={dphy_config.virtual_channel_extension}")

    # C-PHY configuration
    cphy_config = Csi2Config(phy_type=PhyType.CPHY,
                             trio_count=3,
                             bit_rate_mbps=3000,
                             lane_distribution_enabled=True)
    print(
        f"C-PHY: {cphy_config.trio_count} trios @ {cphy_config.bit_rate_mbps} Mbps"
    )
    print(f"       Equivalent lanes: {cphy_config.trio_count * 2}")

    # 2. Packet Creation
    print("\n2. PACKET CREATION")
    print("-" * 20)

    # Short packets
    frame_start = Csi2ShortPacket.frame_start(VirtualChannel.VC1, 1001)
    line_start = Csi2ShortPacket.line_start(VirtualChannel.VC1, 480)
    line_end = Csi2ShortPacket.line_end(VirtualChannel.VC1, 480)
    frame_end = Csi2ShortPacket.frame_end(VirtualChannel.VC1, 1001)

    print(
        f"Frame Start: VC={frame_start.virtual_channel}, Frame={frame_start.data}"
    )
    print(
        f"Line Start:  VC={line_start.virtual_channel}, Line={line_start.data}"
    )
    print(f"Line End:    VC={line_end.virtual_channel}, Line={line_end.data}")
    print(
        f"Frame End:   VC={frame_end.virtual_channel}, Frame={frame_end.data}")

    # Long packets with different data types
    image_data = generate_test_pattern(640, 480, "ramp")
    raw8_packet = Csi2LongPacket(VirtualChannel.VC0, DataType.RAW8, image_data)

    rgb_data = generate_test_pattern(320, 240, "checkerboard")
    rgb_packet = Csi2LongPacket(VirtualChannel.VC2, DataType.RGB888, rgb_data)

    print(
        f"RAW8 Packet:  VC={raw8_packet.virtual_channel}, Size={len(raw8_packet.payload)} bytes"
    )
    print(
        f"RGB888 Packet: VC={rgb_packet.virtual_channel}, Size={len(rgb_packet.payload)} bytes"
    )

    # 3. ECC and Checksum Validation
    print("\n3. ERROR CORRECTION & VALIDATION")
    print("-" * 35)

    data_id = (VirtualChannel.VC0 << 6) | DataType.RAW10
    word_count = 1200
    ecc = calculate_ecc(data_id, word_count)
    print(
        f"ECC Calculation: DI=0x{data_id:02x}, WC={word_count} -> ECC=0x{ecc:02x}"
    )

    test_payload = bytes([i ^ (i << 1) for i in range(100)])  # Test pattern
    checksum = sum(test_payload) & 0xFFFF
    is_valid = validate_checksum(test_payload, checksum)
    print(
        f"Checksum Validation: {len(test_payload)} bytes -> Valid={is_valid}")

    # 4. Data Type Examples
    print("\n4. SUPPORTED DATA TYPES")
    print("-" * 25)

    data_types = [
        DataType.RAW8, DataType.RAW10, DataType.RAW12, DataType.RAW16,
        DataType.RGB888, DataType.RGB565, DataType.RGB444,
        DataType.YUV422_8BIT, DataType.YUV420_8BIT, DataType.FRAME_START,
        DataType.FRAME_END, DataType.LINE_START, DataType.LINE_END
    ]

    for dt in data_types:
        print(f"{dt.name:15}: 0x{dt.value:02x}")

    # 5. Virtual Channel Support
    print("\n5. VIRTUAL CHANNEL SUPPORT")
    print("-" * 28)

    for vc in [VirtualChannel.VC0, VirtualChannel.VC1, VirtualChannel.VC15]:
        packet = Csi2ShortPacket.frame_start(vc, 1)
        print(f"Virtual Channel {vc.value:2d}: 0x{vc.value:01x}")

    # 6. Performance Characteristics
    print("\n6. PERFORMANCE CHARACTERISTICS")
    print("-" * 32)

    configs = [
        ("D-PHY 1-lane 1Gbps",
         Csi2Config(PhyType.DPHY, lane_count=1, bit_rate_mbps=1000)),
        ("D-PHY 4-lane 2.5Gbps",
         Csi2Config(PhyType.DPHY, lane_count=4, bit_rate_mbps=2500)),
        ("C-PHY 1-trio 2Gbps",
         Csi2Config(PhyType.CPHY, trio_count=1, bit_rate_mbps=2000)),
        ("C-PHY 3-trio 3Gbps",
         Csi2Config(PhyType.CPHY, trio_count=3, bit_rate_mbps=3000)),
    ]

    for name, config in configs:
        if config.phy_type == PhyType.DPHY:
            total_bandwidth = config.lane_count * config.bit_rate_mbps
        else:
            total_bandwidth = config.trio_count * config.bit_rate_mbps * 2.28  # C-PHY efficiency

        print(f"{name:20}: {total_bandwidth:.0f} Mbps total bandwidth")

    print("\n=== Demo Complete ===")
    print("\nThe CSI-2 extension provides comprehensive support for:")
    print("• MIPI CSI-2 v4.0.1 specification compliance")
    print("• D-PHY (1-8 lanes) and C-PHY (1-3 trios) physical layers")
    print("• Full packet handling with ECC and checksum validation")
    print("• Multiple data types and virtual channels")
    print("• Error injection for robustness testing")
    print("• Multi-lane configurations with lane distribution")
    print("• Timing validation and performance analysis")


def demonstrate_advanced_features():
    """Demonstrate advanced CSI-2 features"""

    print("\n=== ADVANCED FEATURES ===\n")

    # Error injection configuration
    error_config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=2,
        bit_rate_mbps=1500,
        inject_ecc_errors=True,
        inject_checksum_errors=True,
        error_injection_rate=0.05  # 5% error rate
    )

    print("1. ERROR INJECTION CAPABILITIES")
    print("• ECC error injection for header corruption testing")
    print("• Checksum error injection for payload validation")
    print("• Configurable error rates for stress testing")
    print(
        f"• Example: {error_config.error_injection_rate*100}% error injection rate"
    )

    print("\n2. LANE DISTRIBUTION FEATURES")
    print("• Automatic data distribution across multiple lanes")
    print("• Lane deskew handling for multi-lane synchronization")
    print("• Byte-level interleaving for maximum throughput")

    print("\n3. TIMING VALIDATION")
    print("• PHY layer timing parameter validation")
    print("• HS/LP transition timing checks")
    print("• Clock/data lane synchronization validation")

    print("\n4. PATTERN GENERATION")
    patterns = ["ramp", "checkerboard", "solid", "walking_ones"]
    for pattern in patterns:
        sample = generate_test_pattern(16, 16, pattern)[:8]
        print(f"• {pattern:12}: {[hex(b) for b in sample]}...")


if __name__ == "__main__":
    demonstrate_basic_features()
    demonstrate_advanced_features()

    print("\n" + "=" * 60)
    print("CSI-2 Extension Ready for Integration!")
    print("Use 'from cocotbext.mipi_csi2 import *' to start testing")
    print("=" * 60)
