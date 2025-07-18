"""
MIPI CSI-2 simulation framework for cocotb

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

from .about import __version__
from .bus import Csi2Bus, Csi2DPhyBus, Csi2CPhyBus
from .config import Csi2Config, PhyType
from .packet import Csi2Packet, Csi2ShortPacket, Csi2LongPacket, DataType, VirtualChannel
from .tx import Csi2TxModel
from .rx import Csi2RxModel
from .exceptions import Csi2Exception, Csi2ProtocolError, Csi2PhyError, Csi2EccError

__all__ = [
    "__version__",
    "Csi2Bus",
    "Csi2DPhyBus",
    "Csi2CPhyBus",
    "Csi2Config",
    "PhyType",
    "Csi2Packet",
    "Csi2ShortPacket",
    "Csi2LongPacket",
    "DataType",
    "VirtualChannel",
    "Csi2TxModel",
    "Csi2RxModel",
    "Csi2Exception",
    "Csi2ProtocolError",
    "Csi2PhyError",
    "Csi2EccError"
]
