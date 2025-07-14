"""
MIPI CSI-2 Bus abstraction

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

from typing import Optional, List, Union
from cocotb_bus.bus import Bus
import cocotb
from cocotb.handle import SimHandleBase


class Csi2Bus(Bus):
    """MIPI CSI-2 Bus abstraction supporting both D-PHY and C-PHY interfaces"""

    _signals = ["clk_p", "clk_n", "data_p", "data_n"]
    _optional_signals = ["enable", "reset"]

    def __init__(self, entity: SimHandleBase, prefix: Optional[str] = None, signals=None, **kwargs):
        """
        Initialize CSI-2 bus

        Args:
            entity: The design entity/module
            prefix: Signal prefix
        """
        self.entity = entity
        self.prefix = prefix

        if signals is None:
            signals = self._signals
        # Initialize bus signals
        super().__init__(entity, prefix, signals, optional_signals=self._optional_signals, **kwargs)

        # Additional CSI-2 specific attributes
        self.lane_count = self._detect_lane_count()
        self.phy_type = self._detect_phy_type()

    def _detect_lane_count(self) -> int:
        """Detect number of data lanes from signal names"""
        lane_count = 0

        # Check for data lane signals
        for i in range(8):  # CSI-2 supports up to 8 lanes theoretically
            data_p_name = f"data{i}_p" if not self.prefix else f"{self.prefix}_data{i}_p"
            if hasattr(self.entity, data_p_name):
                lane_count = i + 1

        return max(1, lane_count)  # Default to 1 lane minimum

    def _detect_phy_type(self) -> str:
        """Detect PHY type from signal patterns"""
        # D-PHY has separate _p and _n signals
        # C-PHY has trio signals (A, B, C)

        if hasattr(self.entity, "data0_a") or hasattr(self.entity, f"{self.prefix}_data0_a"):
            return "cphy"
        else:
            return "dphy"

    @classmethod
    def from_entity(cls, entity: SimHandleBase, **kwargs) -> "Csi2Bus":
        """Create CSI-2 bus from entity with automatic signal detection"""
        return cls(entity, **kwargs)

    @classmethod
    def from_prefix(cls, entity: SimHandleBase, prefix: str, **kwargs) -> "Csi2Bus":
        """Create CSI-2 bus from entity with signal prefix"""
        return cls(entity, prefix=prefix, **kwargs)


class Csi2DPhyBus(Csi2Bus):
    """D-PHY specific CSI-2 bus with differential signaling"""

    _signals = ["clk_p", "clk_n"]

    def __init__(self, entity: SimHandleBase, lane_count: int = 1, **kwargs):
        """
        Initialize D-PHY CSI-2 bus

        Args:
            entity: The design entity/module
            lane_count: Number of data lanes (1, 2, 4, 8)
        """
        self.lane_count = lane_count
        signals = ["clk_p", "clk_n"]
        for i in range(lane_count):
            signals.extend([f"data{i}_p", f"data{i}_n"])
        # Only pass signals as a keyword argument, not positionally
        super().__init__(entity, signals=signals, **kwargs)


class Csi2CPhyBus(Csi2Bus):
    """C-PHY specific CSI-2 bus with trio signaling"""

    _signals = []

    def __init__(self, entity: SimHandleBase, trio_count: int = 1, **kwargs):
        """
        Initialize C-PHY CSI-2 bus

        Args:
            entity: The design entity/module
            trio_count: Number of trios (1, 2, 3)
        """
        self.trio_count = trio_count

        # Add trio signals (each trio has A, B, C)
        for i in range(trio_count):
            self._signals.extend([f"trio{i}_a", f"trio{i}_b", f"trio{i}_c"])

        super().__init__(entity, **kwargs)
