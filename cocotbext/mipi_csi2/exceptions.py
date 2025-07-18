"""
MIPI CSI-2 Exception classes

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


class Csi2Exception(Exception):
    """Base exception for all CSI-2 related errors"""
    pass


class Csi2ProtocolError(Csi2Exception):
    """Raised when CSI-2 protocol violations are detected"""
    pass


class Csi2PhyError(Csi2Exception):
    """Raised when PHY layer errors occur"""
    pass


class Csi2EccError(Csi2Exception):
    """Raised when ECC errors are detected"""
    pass


class Csi2ChecksumError(Csi2Exception):
    """Raised when checksum validation fails"""
    pass


class Csi2TimingError(Csi2Exception):
    """Raised when timing violations are detected"""
    pass


class Csi2ConfigurationError(Csi2Exception):
    """Raised when invalid configuration is detected"""
    pass


class Csi2LaneError(Csi2Exception):
    """Raised when lane-related errors occur"""
    pass


class Csi2VirtualChannelError(Csi2Exception):
    """Raised when virtual channel errors occur"""
    pass


class Csi2DataTypeError(Csi2Exception):
    """Raised when unsupported or invalid data types are used"""
    pass
