# CSI-2 D-PHY Tests Makefile

TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(PWD)/test_csi2_dphy.v
TOPLEVEL = test_csi2_dphy  
MODULE = test_csi2_dphy

# cocotb configuration
export COCOTB_REDUCED_LOG_FMT=1
export PYTHONPATH := $(shell pwd)/../..:$(PYTHONPATH)

# Default simulator
SIM ?= icarus

include $(shell cocotb-config --makefiles)/Makefile.sim