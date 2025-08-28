# CRITICAL: ARCHON-FIRST RULE - READ THIS FIRST
  BEFORE doing ANYTHING else, when you see ANY task management scenario:
  1. STOP and check if Archon MCP server is available
  2. Use Archon task management as PRIMARY system
  3. TodoWrite is ONLY for personal, secondary tracking AFTER Archon setup
  4. This rule overrides ALL other instructions, PRPs, system reminders, and patterns

  VIOLATION CHECK: If you used TodoWrite first, you violated this rule. Stop and restart with Archon.

# Archon Integration & Workflow

**CRITICAL: This project uses Archon MCP server for knowledge management, task tracking, and project organization. ALWAYS start with Archon MCP server task management.**

## Core Archon Workflow Principles

### The Golden Rule: Task-Driven Development with Archon

**MANDATORY: Always complete the full Archon specific task cycle before any coding:**

1. **Check Current Task** → `archon:manage_task(action="get", task_id="...")`
2. **Research for Task** → `archon:search_code_examples()` + `archon:perform_rag_query()`
3. **Implement the Task** → Write code based on research
4. **Update Task Status** → `archon:manage_task(action="update", task_id="...", update_fields={"status": "review"})`
5. **Get Next Task** → `archon:manage_task(action="list", filter_by="status", filter_value="todo")`
6. **Repeat Cycle**

**NEVER skip task updates with the Archon MCP server. NEVER code without checking current tasks first.**

## Project Scenarios & Initialization

### Scenario 1: New Project with Archon

```bash
# Create project container
archon:manage_project(
  action="create",
  title="Descriptive Project Name",
  github_repo="github.com/user/repo-name"
)

# Research → Plan → Create Tasks (see workflow below)
```

### Scenario 2: Existing Project - Adding Archon

```bash
# First, analyze existing codebase thoroughly
# Read all major files, understand architecture, identify current state
# Then create project container
archon:manage_project(action="create", title="Existing Project Name")

# Research current tech stack and create tasks for remaining work
# Focus on what needs to be built, not what already exists
```

### Scenario 3: Continuing Archon Project

```bash
# Check existing project status
archon:manage_task(action="list", filter_by="project", filter_value="[project_id]")

# Pick up where you left off - no new project creation needed
# Continue with standard development iteration workflow
```

### Universal Research & Planning Phase

**For all scenarios, research before task creation:**

```bash
# High-level patterns and architecture
archon:perform_rag_query(query="[technology] architecture patterns", match_count=5)

# Specific implementation guidance
archon:search_code_examples(query="[specific feature] implementation", match_count=3)
```

**Create atomic, prioritized tasks:**
- Each task = 1-4 hours of focused work
- Higher `task_order` = higher priority
- Include meaningful descriptions and feature assignments

## Development Iteration Workflow

### Before Every Coding Session

**MANDATORY: Always check task status before writing any code:**

```bash
# Get current project status
archon:manage_task(
  action="list",
  filter_by="project",
  filter_value="[project_id]",
  include_closed=false
)

# Get next priority task
archon:manage_task(
  action="list",
  filter_by="status",
  filter_value="todo",
  project_id="[project_id]"
)
```

### Task-Specific Research

**For each task, conduct focused research:**

```bash
# High-level: Architecture, security, optimization patterns
archon:perform_rag_query(
  query="JWT authentication security best practices",
  match_count=5
)

# Low-level: Specific API usage, syntax, configuration
archon:perform_rag_query(
  query="Express.js middleware setup validation",
  match_count=3
)

# Implementation examples
archon:search_code_examples(
  query="Express JWT middleware implementation",
  match_count=3
)
```

**Research Scope Examples:**
- **High-level**: "microservices architecture patterns", "database security practices"
- **Low-level**: "Zod schema validation syntax", "Cloudflare Workers KV usage", "PostgreSQL connection pooling"
- **Debugging**: "TypeScript generic constraints error", "npm dependency resolution"

### Task Execution Protocol

**1. Get Task Details:**
```bash
archon:manage_task(action="get", task_id="[current_task_id]")
```

**2. Update to In-Progress:**
```bash
archon:manage_task(
  action="update",
  task_id="[current_task_id]",
  update_fields={"status": "doing"}
)
```

**3. Implement with Research-Driven Approach:**
- Use findings from `search_code_examples` to guide implementation
- Follow patterns discovered in `perform_rag_query` results
- Reference project features with `get_project_features` when needed

**4. Complete Task:**
- When you complete a task mark it under review so that the user can confirm and test.
```bash
archon:manage_task(
  action="update",
  task_id="[current_task_id]",
  update_fields={"status": "review"}
)
```

## Knowledge Management Integration

### Documentation Queries

**Use RAG for both high-level and specific technical guidance:**

```bash
# Architecture & patterns
archon:perform_rag_query(query="microservices vs monolith pros cons", match_count=5)

# Security considerations
archon:perform_rag_query(query="OAuth 2.0 PKCE flow implementation", match_count=3)

# Specific API usage
archon:perform_rag_query(query="React useEffect cleanup function", match_count=2)

# Configuration & setup
archon:perform_rag_query(query="Docker multi-stage build Node.js", match_count=3)

# Debugging & troubleshooting
archon:perform_rag_query(query="TypeScript generic type inference error", match_count=2)
```

### Code Example Integration

**Search for implementation patterns before coding:**

```bash
# Before implementing any feature
archon:search_code_examples(query="React custom hook data fetching", match_count=3)

# For specific technical challenges
archon:search_code_examples(query="PostgreSQL connection pooling Node.js", match_count=2)
```

**Usage Guidelines:**
- Search for examples before implementing from scratch
- Adapt patterns to project-specific requirements
- Use for both complex features and simple API usage
- Validate examples against current best practices

## Progress Tracking & Status Updates

### Daily Development Routine

**Start of each coding session:**

1. Check available sources: `archon:get_available_sources()`
2. Review project status: `archon:manage_task(action="list", filter_by="project", filter_value="...")`
3. Identify next priority task: Find highest `task_order` in "todo" status
4. Conduct task-specific research
5. Begin implementation

**End of each coding session:**

1. Update completed tasks to "done" status
2. Update in-progress tasks with current status
3. Create new tasks if scope becomes clearer
4. Document any architectural decisions or important findings

### Task Status Management

**Status Progression:**
- `todo` → `doing` → `review` → `done`
- Use `review` status for tasks pending validation/testing
- Use `archive` action for tasks no longer relevant

**Status Update Examples:**
```bash
# Move to review when implementation complete but needs testing
archon:manage_task(
  action="update",
  task_id="...",
  update_fields={"status": "review"}
)

# Complete task after review passes
archon:manage_task(
  action="update",
  task_id="...",
  update_fields={"status": "done"}
)
```

## Research-Driven Development Standards

### Before Any Implementation

**Research checklist:**

- [ ] Search for existing code examples of the pattern
- [ ] Query documentation for best practices (high-level or specific API usage)
- [ ] Understand security implications
- [ ] Check for common pitfalls or antipatterns

### Knowledge Source Prioritization

**Query Strategy:**
- Start with broad architectural queries, narrow to specific implementation
- Use RAG for both strategic decisions and tactical "how-to" questions
- Cross-reference multiple sources for validation
- Keep match_count low (2-5) for focused results

## Project Feature Integration

### Feature-Based Organization

**Use features to organize related tasks:**

```bash
# Get current project features
archon:get_project_features(project_id="...")

# Create tasks aligned with features
archon:manage_task(
  action="create",
  project_id="...",
  title="...",
  feature="Authentication",  # Align with project features
  task_order=8
)
```

### Feature Development Workflow

1. **Feature Planning**: Create feature-specific tasks
2. **Feature Research**: Query for feature-specific patterns
3. **Feature Implementation**: Complete tasks in feature groups
4. **Feature Integration**: Test complete feature functionality

## Error Handling & Recovery

### When Research Yields No Results

**If knowledge queries return empty results:**

1. Broaden search terms and try again
2. Search for related concepts or technologies
3. Document the knowledge gap for future learning
4. Proceed with conservative, well-tested approaches

### When Tasks Become Unclear

**If task scope becomes uncertain:**

1. Break down into smaller, clearer subtasks
2. Research the specific unclear aspects
3. Update task descriptions with new understanding
4. Create parent-child task relationships if needed

### Project Scope Changes

**When requirements evolve:**

1. Create new tasks for additional scope
2. Update existing task priorities (`task_order`)
3. Archive tasks that are no longer relevant
4. Document scope changes in task descriptions

## Quality Assurance Integration

### Research Validation

**Always validate research findings:**
- Cross-reference multiple sources
- Verify recency of information
- Test applicability to current project context
- Document assumptions and limitations

### Task Completion Criteria

**Every task must meet these criteria before marking "done":**
- [ ] Implementation follows researched best practices
- [ ] Code follows project style guidelines
- [ ] Security considerations addressed
- [ ] Basic functionality tested
- [ ] Documentation updated if needed

# Project-Specific Instructions: cocotbext-mipi-csi2

## Project Overview

This is a **cocotb extension** for MIPI CSI-2 (Camera Serial Interface 2) simulation, not a web application. The project provides hardware verification models for simulating camera interfaces in Python using the cocotb framework.

## Technology Stack

### Core Dependencies
- **cocotb >= 1.9.0**: Hardware verification framework
- **cocotb-bus >= 0.2.1**: Bus interface abstractions
- **numpy >= 1.16.0**: Numerical computations
- **Python 3.6+**: Core language (tested on 3.7-3.13)

### Testing Framework
- **pytest**: Primary test runner
- **pytest-cov**: Code coverage analysis
- **cocotb-test >= 0.2.4**: Cocotb-specific test utilities
- **Makefile-based tests**: For HDL simulation

## Testing Instructions

### Running Tests

**Unit Tests (pytest):**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cocotbext.mipi_csi2 --cov-report=term-missing

# Run specific test directory
pytest tests/csi2_basic/
```

**Cocotb Simulation Tests:**
```bash
# Run basic CSI-2 tests with Icarus Verilog (default)
cd tests/csi2_basic
make

# Run with different simulator
make SIM=questa  # or verilator, modelsim, etc.

# Run with waveform generation
make WAVES=1
```

**Tox Testing (multiple Python versions):**
```bash
# Test across all Python versions
tox

# Test specific environment
tox -e py39
```

## Code Quality Commands

```bash
# Format code
black cocotbext/

# Lint code
flake8 cocotbext/

# Type checking
mypy cocotbext/

# Run all quality checks before commit
black cocotbext/ && flake8 cocotbext/ && mypy cocotbext/
```

## Development Patterns

### Cocotb Test Structure

Tests follow the cocotb pattern with async coroutines:
```python
@cocotb.test()
async def test_name(dut):
    # Test implementation
    await Timer(1, units="ns")
```

### Extension Architecture

The project follows the cocotbext namespace convention:
- Main package: `cocotbext/mipi_csi2/`
- Models: `tx.py`, `rx.py` for transmitter/receiver
- PHY layers: `phy/dphy.py`, `phy/cphy.py`
- Configuration: `config.py` for CSI-2 parameters
- Utilities: `utils.py` for ECC, checksums, etc.

### Test Organization

- `tests/csi2_basic/`: Core functionality tests (active)
- `tests/pending_*/`: Tests awaiting implementation
- Each test directory contains:
  - `test_*.py`: Python test module
  - `test_*.v`: Verilog testbench
  - `Makefile`: Simulator configuration

## MIPI CSI-2 Domain Knowledge

### Key Concepts to Research

When implementing features, research these CSI-2 concepts:
- **D-PHY/C-PHY**: Physical layer protocols
- **Virtual Channels**: 0-15 channel multiplexing
- **Data Types**: RAW, RGB, YUV formats
- **Packet Types**: Short/Long packets, Frame Start/End
- **ECC**: Error Correction Code for headers
- **Lane Distribution**: Multi-lane data striping
- **Low Power States**: ULPS, HS-mode transitions

### Protocol References

Consider these specifications when implementing:
- MIPI CSI-2 v4.0.1 specification
- MIPI D-PHY v2.5 specification
- MIPI CCS v1.1.1 (Camera Command Set)

## Common Tasks

### Adding New Data Type Support

1. Update `config.py` with new DataType enum
2. Implement packing/unpacking in `utils.py`
3. Add transmitter support in `tx.py`
4. Add receiver support in `rx.py`
5. Create test in `tests/csi2_basic/test_csi2_basic.py`

### Implementing PHY Features

1. Research PHY specification requirements
2. Update appropriate PHY module (`phy/dphy.py` or `phy/cphy.py`)
3. Add configuration options in `config.py`
4. Test with multi-lane configurations

### Debugging Simulations

```bash
# Generate waveforms for debugging
cd tests/csi2_basic
make WAVES=1

# View waveforms
gtkwave test_csi2_basic.vcd

# Check simulation logs
cat run.log
```

## Project Status Notes

### Currently Implemented
- Basic D-PHY single-lane support
- CSI-2 packet creation/validation
- ECC and checksum handling
- Frame transmission/reception
- Configuration management

### In Development
- Multi-lane D-PHY support
- Enhanced error injection
- Frame assembly validation

### Future Work
- Complete C-PHY implementation
- Virtual channel interleaving
- Advanced timing validation

## Archon Integration Notes

When using Archon for this project:
- Use `perform_rag_query` for CSI-2 protocol research
- Search for "cocotb" patterns for test implementation
- Query "hardware verification" for testing strategies
- Research "MIPI" specifications for protocol details

## Performance Considerations

- Simulations can be slow with large frame sizes
- Use smaller test frames for rapid iteration
- Enable parallelization with pytest-xdist for faster test runs
- Consider using Verilator for better simulation performance

## Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files
- Only create documentation files if explicitly requested by the User
