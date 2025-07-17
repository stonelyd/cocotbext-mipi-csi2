# MIPI CSI-2 simulation framework for Cocotb

[![Regression Tests](https://github.com/stonelyd/cocotbext-mipi-csi2/actions/workflows/regression-tests.yml/badge.svg)](https://github.com/stonelyd/cocotbext-mipi-csi2/actions/workflows/regression-tests.yml)
[![codecov](https://codecov.io/gh/stonelyd/cocotbext-mipi-csi2/branch/master/graph/badge.svg)](https://codecov.io/gh/stonelyd/cocotbext-mipi-csi2)
[![PyPI version](https://badge.fury.io/py/cocotbext-mipi-csi2.svg)](https://pypi.org/project/cocotbext-mipi-csi2)
[![Downloads](https://pepy.tech/badge/cocotbext-mipi-csi2)](https://pepy.tech/project/cocotbext-mipi-csi2)

GitHub repository: https://github.com/stonelyd/cocotbext-mipi-csi2


Note: This project is in active developmnet, many of the features listed below have not be implamented or tested.

## Introduction

MIPI CSI-2 (Camera Serial Interface 2) simulation framework for [cocotb](https://github.com/cocotb/cocotb).

This package provides comprehensive simulation models for MIPI CSI-2 protocol, supporting both D-PHY and C-PHY physical layers. It includes transmitter and receiver models for testing CSI-2 implementations with extensive error injection and validation capabilities.

## Features

### Protocol Support
- **CSI-2 v4.0.1 compliant** implementation
- **D-PHY** physical layer support (1, 2, 4 lanes)
- **C-PHY** physical layer support (1, 2, 3 trios)
- **Virtual Channel** support (0-15)
- **Multiple data types**: RAW8/10/12/16, RGB888/565, YUV422/420

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

### Advanced Features
- **Continuous streaming** simulation
- **Multi-virtual channel** concurrent transmission
- **Lane deskew** handling
- **Scrambling** support (CSI-2 v2.0+)
- **Extended virtual channels** (CSI-2 v2.0+)

## Installation

Installation from pip (release version, stable):

    $ pip install cocotbext-mipi-csi2


Installation for active development:

    $ git clone https://github.com/stonelyd/cocotbext-mipi-csi2
    $ pip install -e cocotbext-mipi-csi2

