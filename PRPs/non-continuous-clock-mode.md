# Non-Continuous Clock Mode Enhancement PRP

## Goal
Implement MIPI D-PHY non-continuous clock mode operation in cocotbext-mipi-csi2, enabling the clock lane to enter Low-Power (LP-11) state between packet transmissions for power savings while maintaining full backward compatibility with existing continuous clock mode.

## Why
- **Power Efficiency**: Reduces power consumption by allowing clock lane to enter LP mode between transmissions
- **CSI-2 Compliance**: Implements CSI-2 specification Section 5.5.2.2 non-continuous clock mode requirements  
- **Mobile/Battery Applications**: Critical for battery-powered camera applications requiring low power operation
- **Framework Completeness**: Fills key gap in framework's CSI-2 compliance and feature completeness

## What
Non-continuous clock mode allows the D-PHY clock lane to transition between High-Speed (HS) and Low-Power (LP-11) states based on data transmission activity, controlled by the existing `continuous_clock` configuration flag.

### Success Criteria
- [ ] Clock lane properly transitions LP-11 ↔ HS-0 ↔ HS-1 with correct D-PHY timing
- [ ] Data transmission coordinates with clock lane state transitions
- [ ] Receiver correctly handles both continuous and non-continuous clock patterns
- [ ] All existing continuous clock mode tests pass unchanged
- [ ] New non-continuous mode tests achieve >95% code coverage
- [ ] Performance overhead <10% compared to continuous mode

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- spec: D-PHY v2.5 Specification Section 6.10
  why: Clock lane timing parameters (t_clk_prepare, t_clk_zero, t_clk_pre, t_clk_post, t_clk_trail)
  critical: LP-11 → LP-01 → LP-00 → HS-0 → HS-1 timing sequences
  
- spec: CSI-2 v4.0.1 Specification Section 5.5.2.2  
  why: Non-continuous clock mode operational requirements
  critical: Clock pre/post amble timing around data transmission
  
- file: cocotbext/mipi_csi2/phy/dphy.py:195-197
  why: Current continuous clock generation implementation
  pattern: _generate_continuous_clock() method shows clock generation approach
  
- file: cocotbext/mipi_csi2/phy/dphy.py:543-606
  why: Existing packet transmission coordination with timing
  pattern: start_packet_transmission() and stop_packet_transmission() methods
  
- file: cocotbext/mipi_csi2/phy/dphy.py:79-101  
  why: D-PHY lane state definitions and management
  critical: DPhyState.LP_00, LP_01, LP_11, HS_0, HS_1 states already defined
  
- file: cocotbext/mipi_csi2/config.py:140
  why: continuous_clock boolean flag controls behavior
  pattern: Existing configuration approach to maintain backward compatibility
```

### Current Codebase Tree (relevant files)
```bash
cocotbext/mipi_csi2/
├── config.py                  # Csi2Config with continuous_clock flag
├── phy/
│   ├── dphy.py                # DPhyTxModel and DPhyRxModel - MAIN TARGET
│   └── __init__.py
├── tx.py                      # High-level transmitter using DPhyTxModel  
├── rx.py                      # High-level receiver using DPhyRxModel
└── utils.py                   # Utility functions
tests/csi2_basic/
├── test_csi2_basic.py         # Main test patterns - MODEL FOR NEW TESTS
└── test_csi2_basic.v          # Verilog testbench
```

### Desired Codebase Tree (no new files needed)
```bash
# NO NEW FILES - All changes are enhancements to existing files
cocotbext/mipi_csi2/phy/dphy.py  # Enhanced with non-continuous clock methods
tests/csi2_basic/test_csi2_basic.py  # New test cases for non-continuous mode
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: cocotb timing requires proper async/await for all Timer() calls
# Example: await Timer(self.phy_config.t_clk_prepare, units='ns')

# CRITICAL: D-PHY clock lane state management already exists in DPhyTxModel
# Lane states: self.lane_states[lane.name] = DPhyState.LP_11 (line 206)
# Signal access: self.lane_signals[lane.name]['p'] and ['n'] (line 222)

# CRITICAL: Backward compatibility requirement
# When continuous_clock=True, existing _generate_continuous_clock() MUST remain unchanged
# When continuous_clock=False, NEW non-continuous methods are used

# CRITICAL: Multi-lane coordination  
# Clock lane coordinates with ALL data lanes during start/stop transmission
# See lines 547-605 for existing coordination pattern

# CRITICAL: Timing parameter validation
# Csi2PhyConfig.validate_timing() must pass for all timing parameters (line 131)
# UI (Unit Interval) = 1000.0 / bit_rate_mbps nanoseconds

# PATTERN: Signal setting in D-PHY
# LP states: signals['p'].value = X, signals['n'].value = Y  
# HS states: Use differential signaling (p=1,n=0 for HS-1; p=0,n=1 for HS-0)

# GOTCHA: cocotb concurrent execution
# Use cocotb.start_soon() for concurrent clock generation tasks
# Current continuous clock: self._clock_task = cocotb.start_soon(self._generate_continuous_clock())
```

## Implementation Blueprint

### Core Methods to Add to DPhyTxModel class

1. **Clock Lane State Management Infrastructure**
```python
async def _start_non_continuous_clock(self):
    """Start non-continuous clock sequence: LP-11 → LP-01 → LP-00 → HS-0 → HS-1"""
    # PATTERN: Follow existing _hs_prepare_sequence() structure (lines 418-446)
    
async def _stop_non_continuous_clock(self):  
    """Stop non-continuous clock sequence: HS-1 → HS-0 → LP-11"""
    # PATTERN: Follow existing _hs_exit_sequence() structure (lines 447-465)
    
def _clock_lane_hs_sequence(self, start: bool):
    """Coordinate clock lane HS entry/exit with proper timing"""
    # CRITICAL: Use self.phy_config timing parameters
```

### List of Tasks (Implementation Order)

```yaml
Task 1: Enhance DPhyTxModel with Non-Continuous Clock Infrastructure
MODIFY cocotbext/mipi_csi2/phy/dphy.py:
  - FIND pattern: "def __init__(self, bus: Csi2DPhyBus, config: Csi2Config):" (line 106)
  - PRESERVE existing continuous clock initialization (lines 195-197)
  - ADD clock lane state tracking variables
  - KEEP all existing functionality intact

Task 2: Implement Clock Lane HS Entry Sequence  
MODIFY cocotbext/mipi_csi2/phy/dphy.py:
  - ADD async def _start_non_continuous_clock(self): method
  - MIRROR timing pattern from _hs_prepare_sequence() (line 418)
  - APPLY D-PHY timing: t_clk_prepare, t_clk_zero, t_clk_pre from self.phy_config
  - COORDINATE with existing clock lane signal access pattern

Task 3: Implement Clock Lane HS Exit Sequence
MODIFY cocotbext/mipi_csi2/phy/dphy.py:  
  - ADD async def _stop_non_continuous_clock(self): method
  - MIRROR timing pattern from _hs_exit_sequence() (line 447)
  - APPLY D-PHY timing: t_clk_post, t_clk_trail from self.phy_config
  - ENSURE clock lane returns to LP-11 state

Task 4: Coordinate Packet Transmission with Non-Continuous Clock
MODIFY cocotbext/mipi_csi2/phy/dphy.py:
  - FIND start_packet_transmission() method (line 543)
  - INJECT non-continuous clock start before data lane HS prepare
  - FIND stop_packet_transmission() method (line 583)  
  - INJECT non-continuous clock stop after data lane HS exit
  - PRESERVE existing continuous clock behavior when config.continuous_clock=True

Task 5: Enhance DPhyRxModel for Non-Continuous Clock Detection
MODIFY cocotbext/mipi_csi2/phy/dphy.py:
  - FIND _monitor_clock_events() method (line 699)
  - ENHANCE to handle LP-11 ↔ HS transitions on clock lane
  - PRESERVE existing data lane sampling logic
  - ADD clock pattern detection for both modes

Task 6: Add Comprehensive Test Cases
MODIFY tests/csi2_basic/test_csi2_basic.py:
  - MIRROR existing test pattern structure from run_short_packet_transmission (line 100)
  - CREATE test_non_continuous_short_packet() function
  - CREATE test_non_continuous_long_packet() function  
  - CREATE test_non_continuous_frame_transmission() function
  - VALIDATE timing compliance and backward compatibility
```

### Per Task Pseudocode

```python
# Task 1: Clock Infrastructure  
def __init__(self, bus, config):
    # ... existing initialization ...
    
    # NEW: Non-continuous clock state tracking
    self.clock_lane_hs_active = False
    self.non_continuous_clock_task = None
    
    # PRESERVE: Existing continuous clock logic
    if config.continuous_clock:
        self._clock_task = cocotb.start_soon(self._generate_continuous_clock())

# Task 2: Clock HS Entry
async def _start_non_continuous_clock(self):
    """LP-11 → LP-01 → LP-00 → HS-0 → HS-1"""
    clock_signals = self.lane_signals[self.clock_lane.name]
    
    # CRITICAL: Follow D-PHY v2.5 Section 6.10 timing
    # Step 1: LP-01 state  
    clock_signals['p'].value = 0
    clock_signals['n'].value = 1
    await Timer(self.phy_config.t_lpx, units='ns')
    
    # Step 2: LP-00 (Bridge state)
    clock_signals['p'].value = 0  
    clock_signals['n'].value = 0
    await Timer(self.phy_config.t_clk_prepare, units='ns')
    
    # Step 3: HS-0 state
    clock_signals['p'].value = 0
    clock_signals['n'].value = 1
    await Timer(self.phy_config.t_clk_zero, units='ns')
    
    # Step 4: Start HS clock generation
    self.clock_lane_hs_active = True
    self.non_continuous_clock_task = cocotb.start_soon(self._generate_hs_clock_burst())

# Task 3: Clock HS Exit  
async def _stop_non_continuous_clock(self):
    """HS-1 → HS-0 → LP-11"""
    # Stop HS clock generation
    if self.non_continuous_clock_task:
        self.non_continuous_clock_task.kill()
        self.non_continuous_clock_task = None
    
    clock_signals = self.lane_signals[self.clock_lane.name]
    
    # CRITICAL: Apply t_clk_post timing before trail
    await Timer(self.phy_config.t_clk_post, units='ns')
    
    # Drive HS-0 during trail period
    clock_signals['p'].value = 0
    clock_signals['n'].value = 1
    await Timer(self.phy_config.t_clk_trail, units='ns')
    
    # Return to LP-11
    clock_signals['p'].value = 1
    clock_signals['n'].value = 1
    self.clock_lane_hs_active = False

# Task 4: Packet Coordination
async def start_packet_transmission(self):
    """Enhanced with non-continuous clock coordination"""
    self.logger.info("TX PHY: Starting packet transmission with non-continuous clock support")
    
    # NEW: Non-continuous clock HS entry
    if not self.config.continuous_clock:
        await self._start_non_continuous_clock()
        # CRITICAL: Wait for t_clk_pre before data lane HS prepare
        await Timer(self.phy_config.t_clk_pre, units='ns') 
    
    # PRESERVE: Existing data lane HS prepare sequence (lines 550-581)
    # ... existing data lane logic unchanged ...
```

### Integration Points
```yaml
CONFIGURATION:
  - file: cocotbext/mipi_csi2/config.py  
  - existing: continuous_clock boolean flag (line 140)
  - pattern: "continuous_clock: bool = False  # Non-continuous is default per spec"
  
TIMING_VALIDATION:
  - file: cocotbext/mipi_csi2/config.py
  - existing: Csi2PhyConfig.validate_timing() method (line 249)
  - enhancement: Add non-continuous clock timing validation
  
TEST_INTEGRATION:
  - file: tests/csi2_basic/test_csi2_basic.py
  - pattern: TestFactory.add_option() for parameterized tests (line 430)
  - add: "continuous_clock" parameter [True, False] to existing test factories
```

## Validation Loop

### Level 1: Syntax & Style  
```bash
# Run these FIRST - fix any errors before proceeding
ruff check cocotbext/mipi_csi2/phy/dphy.py --fix
mypy cocotbext/mipi_csi2/phy/dphy.py

# Expected: No errors. D-PHY module must pass type checking.
```

### Level 2: Unit Tests - Clock Lane Behavior
```python
# ADD to test_csi2_basic.py - New test cases for non-continuous mode
@cocotb.test()
async def test_non_continuous_clock_timing(dut):
    """Validate D-PHY clock lane timing compliance"""
    config = Csi2Config(continuous_clock=False, bit_rate_mbps=800)
    # Test t_clk_prepare, t_clk_zero, t_clk_pre, t_clk_post, t_clk_trail timing
    
async def test_non_continuous_short_packet_1_lane(dut):
    """Test non-continuous mode with single lane short packet"""
    # MIRROR: run_short_packet_transmission pattern
    # SET: continuous_clock=False in config
    
async def test_non_continuous_long_packet_4_lane(dut):  
    """Test non-continuous mode with multi-lane long packet"""
    # MIRROR: run_long_packet_transmission pattern
    # SET: continuous_clock=False, lane_count=4
    
async def test_backward_compatibility(dut):
    """Ensure continuous_clock=True behavior unchanged"""  
    # RUN: Existing tests with continuous_clock=True
    # ASSERT: Identical behavior to baseline
```

```bash
# Run cocotb tests with specific non-continuous patterns:
cd tests/csi2_basic
make WAVES=1  # Generate waveforms for timing validation
pytest test_csi2_basic.py::test_non_continuous_clock_timing -v -s

# Expected: All non-continuous clock tests pass
# If failing: Check timing parameters against D-PHY spec requirements
```

### Level 3: Integration Test - Multi-Mode Support
```bash
# Test both continuous and non-continuous modes:
cd tests/csi2_basic  
make SIM=icarus  # Run full test suite

# Validate waveforms show proper clock lane behavior:
gtkwave test_csi2_basic.vcd
# Check: Clock lane LP-11 periods between packets (non-continuous)
# Check: Clock lane continuous HS toggle (continuous mode)

# Expected: Both modes work correctly with proper timing
```

## Final Validation Checklist
- [ ] All existing tests pass: `pytest tests/csi2_basic/ -v`
- [ ] New non-continuous tests pass: `pytest -k "non_continuous" -v`
- [ ] No linting errors: `ruff check cocotbext/mipi_csi2/phy/dphy.py`
- [ ] No type errors: `mypy cocotbext/mipi_csi2/phy/dphy.py`
- [ ] Timing validation: Clock lane waveforms show correct LP-11 ↔ HS transitions
- [ ] Performance test: <10% overhead compared to continuous mode
- [ ] Backward compatibility: continuous_clock=True behavior identical to baseline

---

## Anti-Patterns to Avoid
- ❌ Don't break existing continuous clock behavior - it's heavily tested
- ❌ Don't ignore D-PHY timing requirements - they're mandatory for compliance  
- ❌ Don't add new configuration flags - use existing continuous_clock boolean
- ❌ Don't modify existing method signatures - maintain API compatibility
- ❌ Don't skip timing validation - cocotb Timer() calls must use proper units
- ❌ Don't forget multi-lane coordination - clock lane affects all data lanes
- ❌ Don't hardcode timing values - use self.phy_config parameters

## Confidence Score: 9/10
This PRP provides comprehensive context including:
- ✅ Complete D-PHY specification references with specific timing requirements
- ✅ Detailed codebase analysis with exact line numbers and patterns to follow  
- ✅ Step-by-step implementation tasks with pseudocode and timing details
- ✅ Comprehensive test strategy mirroring existing patterns
- ✅ Backward compatibility preservation approach
- ✅ Executable validation steps with expected outcomes

The implementation should succeed in one pass with this level of context and validation loops.