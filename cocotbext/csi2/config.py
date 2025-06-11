"""
MIPI CSI-2 Configuration classes

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

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Dict, Any


class PhyType(Enum):
    """PHY layer types supported by CSI-2"""
    DPHY = "dphy"
    CPHY = "cphy"


class DataType(IntEnum):
    """CSI-2 Data Types according to specification"""
    # Synchronization Short Packet Data Types
    FRAME_START = 0x00
    FRAME_END = 0x01
    LINE_START = 0x02
    LINE_END = 0x03
    
    # Generic Short Packet Data Types  
    GENERIC_SHORT_PACKET_1 = 0x08
    GENERIC_SHORT_PACKET_2 = 0x09
    GENERIC_SHORT_PACKET_3 = 0x0A
    GENERIC_SHORT_PACKET_4 = 0x0B
    GENERIC_SHORT_PACKET_5 = 0x0C
    GENERIC_SHORT_PACKET_6 = 0x0D
    GENERIC_SHORT_PACKET_7 = 0x0E
    GENERIC_SHORT_PACKET_8 = 0x0F
    
    # YUV Data Types
    YUV420_8BIT = 0x18
    YUV420_10BIT = 0x19
    YUV420_8BIT_LEGACY = 0x1A
    YUV420_8BIT_CHROMA_SHIFTED = 0x1C
    YUV420_10BIT_CHROMA_SHIFTED = 0x1D
    YUV422_8BIT = 0x1E
    YUV422_10BIT = 0x1F
    
    # RGB Data Types
    RGB444 = 0x20
    RGB555 = 0x21
    RGB565 = 0x22
    RGB666 = 0x23
    RGB888 = 0x24
    
    # RAW Data Types
    RAW6 = 0x28
    RAW7 = 0x29
    RAW8 = 0x2A
    RAW10 = 0x2B
    RAW12 = 0x2C
    RAW14 = 0x2D
    RAW16 = 0x2E
    RAW20 = 0x2F
    
    # User Defined Types
    USER_DEFINED_1 = 0x30
    USER_DEFINED_2 = 0x31
    USER_DEFINED_3 = 0x32
    USER_DEFINED_4 = 0x33
    USER_DEFINED_5 = 0x34
    USER_DEFINED_6 = 0x35
    USER_DEFINED_7 = 0x36
    USER_DEFINED_8 = 0x37
    
    # Generic Long Packet Data Types
    GENERIC_LONG_PACKET_1 = 0x10
    GENERIC_LONG_PACKET_2 = 0x11
    GENERIC_LONG_PACKET_3 = 0x12
    GENERIC_LONG_PACKET_4 = 0x13


class VirtualChannel(IntEnum):
    """Virtual Channel identifiers (0-3 for CSI-2 v1.x, 0-15 for v2.x+)"""
    VC0 = 0
    VC1 = 1
    VC2 = 2
    VC3 = 3
    VC4 = 4
    VC5 = 5
    VC6 = 6
    VC7 = 7
    VC8 = 8
    VC9 = 9
    VC10 = 10
    VC11 = 11
    VC12 = 12
    VC13 = 13
    VC14 = 14
    VC15 = 15


@dataclass
class Csi2Config:
    """CSI-2 Configuration parameters"""
    
    # Physical layer configuration
    phy_type: PhyType = PhyType.DPHY
    lane_count: int = 1  # 1, 2, 4, 8 for D-PHY; derived from trio_count for C-PHY
    trio_count: int = 1  # For C-PHY: 1, 2, 3 trios
    
    # Timing configuration
    bit_rate_mbps: float = 800.0  # Data rate in Mbps per lane
    t_clk_prepare_ns: float = 38.0  # Clock prepare time
    t_clk_zero_ns: float = 262.0   # Clock zero time
    t_clk_pre_ns: float = 8.0      # Clock pre time
    t_clk_post_ns: float = 60.0    # Clock post time
    t_clk_trail_ns: float = 60.0   # Clock trail time
    
    t_hs_prepare_ns: float = 40.0  # HS prepare time
    t_hs_zero_ns: float = 105.0    # HS zero time
    t_hs_trail_ns: float = 60.0    # HS trail time
    t_hs_exit_ns: float = 100.0    # HS exit time
    
    # Protocol configuration
    continuous_clock: bool = False  # Continuous vs. non-continuous clock
    scrambling_enabled: bool = False  # CSI-2 v2.0+ scrambling
    virtual_channel_extension: bool = False  # Extended VC support (v2.0+)
    
    # Packet configuration
    max_packet_size: int = 65535  # Maximum long packet size
    enable_ecc: bool = True       # Error correction code
    enable_checksum: bool = True  # Checksum for long packets
    
    # Lane distribution configuration
    lane_distribution_enabled: bool = True
    lane_interleaving: bool = False
    
    # Debug and test configuration
    inject_ecc_errors: bool = False
    inject_checksum_errors: bool = False
    inject_crc_errors: bool = False
    error_injection_rate: float = 0.0  # 0.0 to 1.0
    
    # Simulation specific
    clock_period_ns: float = 10.0  # System clock period
    reset_active_level: int = 1    # Reset polarity
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.phy_type == PhyType.DPHY:
            if self.lane_count not in [1, 2, 4, 8]:
                raise ValueError(f"Invalid D-PHY lane count: {self.lane_count}")
        elif self.phy_type == PhyType.CPHY:
            if self.trio_count not in [1, 2, 3]:
                raise ValueError(f"Invalid C-PHY trio count: {self.trio_count}")
            # For C-PHY, lane count is derived from trio count
            self.lane_count = self.trio_count * 3
            
        if self.bit_rate_mbps <= 0:
            raise ValueError(f"Invalid bit rate: {self.bit_rate_mbps}")
            
        if not 0.0 <= self.error_injection_rate <= 1.0:
            raise ValueError(f"Invalid error injection rate: {self.error_injection_rate}")
    
    def get_bit_period_ns(self) -> float:
        """Calculate bit period in nanoseconds"""
        return 1000.0 / self.bit_rate_mbps
    
    def get_byte_period_ns(self) -> float:
        """Calculate byte period in nanoseconds"""
        return self.get_bit_period_ns() * 8
    
    def get_ui_period_ns(self) -> float:
        """Get Unit Interval period in nanoseconds"""
        return self.get_bit_period_ns()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'phy_type': self.phy_type.value,
            'lane_count': self.lane_count,
            'trio_count': self.trio_count,
            'bit_rate_mbps': self.bit_rate_mbps,
            'continuous_clock': self.continuous_clock,
            'scrambling_enabled': self.scrambling_enabled,
            'virtual_channel_extension': self.virtual_channel_extension,
            'enable_ecc': self.enable_ecc,
            'enable_checksum': self.enable_checksum,
            'lane_distribution_enabled': self.lane_distribution_enabled,
            'error_injection_rate': self.error_injection_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Csi2Config":
        """Create configuration from dictionary"""
        # Convert phy_type string back to enum
        if 'phy_type' in config_dict:
            config_dict['phy_type'] = PhyType(config_dict['phy_type'])
        
        return cls(**config_dict)


@dataclass  
class Csi2PhyConfig:
    """Physical layer specific configuration"""
    
    # D-PHY specific timing parameters (all in nanoseconds)
    t_lpx: float = 50.0           # LP-TX transition time
    t_clk_prepare: float = 38.0   # Clock lane prepare time
    t_clk_zero: float = 262.0     # Clock lane zero time  
    t_clk_pre: float = 8.0        # Clock pre time
    t_clk_post: float = 60.0      # Clock post time
    t_clk_trail: float = 60.0     # Clock trail time
    t_hs_prepare: float = 40.0    # Data lane HS prepare time
    t_hs_zero: float = 105.0      # Data lane HS zero time
    t_hs_trail: float = 60.0      # Data lane HS trail time
    t_hs_exit: float = 100.0      # HS exit time
    t_hs_sync: float = 4.0        # HS sync time
    t_hs_skip: float = 40.0       # HS skip time
    t_hs_settle: float = 85.0     # HS settle time
    t_ta_go: float = 4.0          # Turn-around go time
    t_ta_sure: float = 1.0        # Turn-around sure time
    t_ta_get: float = 5.0         # Turn-around get time
    
    # C-PHY specific parameters
    t_3phase_prepare: float = 38.0  # 3-phase prepare time
    t_3phase_zero: float = 262.0    # 3-phase zero time
    t_3phase_post: float = 60.0     # 3-phase post time
    
    # Common parameters
    voltage_swing_mv: float = 200.0  # Differential voltage swing
    common_mode_voltage_mv: float = 200.0  # Common mode voltage
    
    def validate_timing(self, bit_rate_mbps: float) -> bool:
        """Validate timing parameters against bit rate"""
        ui_ns = 1000.0 / bit_rate_mbps  # Unit interval in ns
        
        # Check minimum timing requirements
        if self.t_clk_prepare < 38.0:
            return False
        if self.t_hs_prepare < 40.0 + 4 * ui_ns:
            return False
        if self.t_hs_zero < 105.0 + 6 * ui_ns:
            return False
            
        return True
