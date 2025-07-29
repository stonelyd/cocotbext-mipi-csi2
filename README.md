# MIPI CSI-2 simulation framework for Cocotb

[![Regression Tests](https://github.com/stonelyd/cocotbext-mipi-csi2/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/stonelyd/cocotbext-mipi-csi2/actions/workflows/regression-tests.yml)
[![codecov](https://codecov.io/gh/stonelyd/cocotbext-mipi-csi2/graph/badge.svg?token=GZA2X439U8)](https://codecov.io/gh/stonelyd/cocotbext-mipi-csi2)
[![PyPI version](https://badge.fury.io/py/cocotbext-mipi-csi2.svg)](https://pypi.org/project/cocotbext-mipi-csi2)
[![Downloads](https://pepy.tech/badge/cocotbext-mipi-csi2)](https://pepy.tech/project/cocotbext-mipi-csi2)

GitHub repository: https://github.com/stonelyd/cocotbext-mipi-csi2

## Introduction

MIPI CSI-2 (Camera Serial Interface 2) simulation framework for [cocotb](https://github.com/cocotb/cocotb).

This package provides comprehensive simulation models for MIPI CSI-2 protocol, supporting both D-PHY and C-PHY physical layers. It includes transmitter and receiver models for testing CSI-2 implementations with extensive error injection and validation capabilities.

## Features

### Protocol Support
- **CSI-2 v4.0.1 compliant** implementation
- **D-PHY** physical layer support (1, 2, 4)
- **Virtual Channel** support (0-15)
- **Multiple data types**: RAW6/7/8/10/12/14/16/20, RGB444/555/565/666/888, YUV420/422, Generic packets

### Packet Handling
- **Short packets**: Frame/Line Start/End, Generic short packets
- **Long packets**: Image data, Generic long packets
- **Error Correction Code (ECC)** generation and validation
- **Checksum** calculation and verification
- **Lane distribution** and merging for multi-lane configurations

### Testing Capabilities
- **Error injection**: ECC errors, checksum errors, timing violations
- **Frame assembly** and validation
- **Timing validation** with configurable parameters
- **Performance analysis** and throughput measurement
- **Pattern generation** for testing (ramp, checkerboard, solid)

### Current Development Status
- **Basic CSI-2 functionality**: Complete and tested
- **D-PHY single-lane support**: Implemented and tested
- **C-PHY implementation**: Future Work
- **Testing Capabilities**: Comming Soon

## Installation

Installation from pip (release version, stable):

    $ pip install cocotbext-mipi-csi2

Installation for active development:

    $ git clone https://github.com/stonelyd/cocotbext-mipi-csi2
    $ pip install -e cocotbext-mipi-csi2

## Quick Start

```python
import cocotb
from cocotbext.mipi_csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config,
    PhyType, DataType, VirtualChannel
)

# Configure CSI-2 interface
config = Csi2Config(
    phy_type=PhyType.DPHY,
    lane_count=2,
    bit_rate_mbps=800.0
)

# Create transmitter and receiver models
tx_model = Csi2TxModel(bus, config)
rx_model = Csi2RxModel(bus, config)

# Send a test frame
await tx_model.send_frame(
    width=640, height=480,
    data_type=DataType.RAW8,
    virtual_channel=0
)

# Receive and validate frame
frame = await rx_model.get_next_frame()
```

## Upcoming Features

The following features are planned for future releases:

1. **Enhanced Testing Capabilities** - Comprehensive test coverage for all data types and configurations
2. **Data Type Transition Tests** - Frame transitions, Virtual Channel interleave, Data Type interleave testing
3. **Complete C-PHY Support** - Full C-PHY implementation with comprehensive testing
4. **Advanced Error Correction** - ECC error correction capabilities
5. **Performance Optimization** - Improved throughput and timing accuracy

<!-- ## Contributing

Contributions are welcome! Please see the [Contributing Guidelines](CONTRIBUTING.md) for details. -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

