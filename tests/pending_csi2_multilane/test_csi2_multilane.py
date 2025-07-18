"""
CSI-2 Multi-lane and performance tests

Copyright (c) 2024 CSI-2 Extension Contributors
"""

import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
import pytest
import asyncio

from cocotbext.mipi_csi2 import (
    Csi2TxModel, Csi2RxModel, Csi2Config, PhyType, DataType,
    Csi2DPhyBus, Csi2CPhyBus, Csi2ShortPacket, Csi2LongPacket,
    Csi2ImageTransmitter, Csi2ImageReceiver
)


@cocotb.test()
async def test_multilane_dphy_performance(dut):
    """Test multi-lane D-PHY performance scaling"""

    # Create clock
    clock = Clock(dut.clk, 8, units="ns")  # 125 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(200, units="ns")
    dut.reset_n.value = 1
    await Timer(200, units="ns")

    # Test different lane configurations
    lane_configs = [1, 2, 4, 8]
    performance_results = {}

    for lanes in lane_configs:
        if lanes > 4:  # Skip if hardware doesn't support
            continue

        cocotb.log.info(f"Testing {lanes}-lane D-PHY configuration")

        # Configure D-PHY
        config = Csi2Config(
            phy_type=PhyType.DPHY,
            lane_count=lanes,
            bit_rate_mbps=1000,
            continuous_clock=True,
            lane_distribution_enabled=True
        )

        # Create bus and models
        bus = Csi2DPhyBus(dut, lane_count=lanes)
        tx_model = Csi2TxModel(bus, config)
        rx_model = Csi2RxModel(bus, config)

        # Test frame transmission
        width, height = 640, 480
        start_time = cocotb.utils.get_sim_time('ns')

        await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
        frame_received = await rx_model.wait_for_frame(0, timeout_ns=200000)

        end_time = cocotb.utils.get_sim_time('ns')

        assert frame_received, f"{lanes}-lane frame not received"

        # Calculate performance
        transmission_time = end_time - start_time
        frame_size = width * height
        throughput_mbps = (frame_size * 8) / (transmission_time / 1000)

        performance_results[lanes] = {
            'time_ns': transmission_time,
            'throughput_mbps': throughput_mbps
        }

        cocotb.log.info(f"{lanes}-lane: {transmission_time}ns, {throughput_mbps:.1f} Mbps")

        # Reset for next test
        await tx_model.reset()
        await rx_model.reset()
        await Timer(1000, units="ns")

    # Verify performance scaling
    if len(performance_results) > 1:
        base_lanes = min(performance_results.keys())
        base_throughput = performance_results[base_lanes]['throughput_mbps']

        for lanes in performance_results:
            if lanes > base_lanes:
                throughput = performance_results[lanes]['throughput_mbps']
                scaling_factor = throughput / base_throughput
                expected_factor = lanes / base_lanes

                cocotb.log.info(f"Scaling {base_lanes} to {lanes} lanes: "
                               f"{scaling_factor:.2f}x (expected ~{expected_factor:.2f}x)")

    cocotb.log.info("Multi-lane D-PHY performance test completed")


@cocotb.test()
async def test_multilane_data_integrity(dut):
    """Test data integrity across multiple lanes"""

    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Configure 4-lane D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=1500,
        continuous_clock=True,
        lane_distribution_enabled=True
    )

    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)

    # Test with different data patterns
    test_patterns = [
        ("sequential", bytes(range(256))),
        ("alternating", bytes([0xAA, 0x55] * 128)),
        ("walking_ones", bytes([(1 << (i % 8)) for i in range(256)])),
        ("random_like", bytes([i ^ (i << 1) ^ (i << 2) for i in range(256)]))
    ]

    for pattern_name, pattern_data in test_patterns:
        cocotb.log.info(f"Testing pattern: {pattern_name}")

        # Send pattern as long packet
        packet = Csi2LongPacket(0, DataType.RAW8, pattern_data)
        await tx_model.send_packet(packet)

        # Receive and verify
        received_packet = await rx_model.get_next_packet(timeout_ns=10000)
        assert received_packet is not None, f"Pattern {pattern_name} not received"
        assert isinstance(received_packet, Csi2LongPacket)
        assert received_packet.payload == pattern_data, \
            f"Data integrity failure for pattern {pattern_name}"

        cocotb.log.info(f"Pattern {pattern_name}: PASS")

    cocotb.log.info("Multi-lane data integrity test completed")


@cocotb.test()
async def test_multilane_virtual_channels(dut):
    """Test multiple virtual channels with multi-lane configuration"""

    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Configure multi-lane D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=2000,
        continuous_clock=True,
        lane_distribution_enabled=True
    )

    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)

    # Send concurrent frames on different VCs
    frame_tasks = []
    vc_configs = [
        {'vc': 0, 'width': 640, 'height': 480, 'dt': DataType.RAW8},
        {'vc': 1, 'width': 320, 'height': 240, 'dt': DataType.RAW10},
        {'vc': 2, 'width': 160, 'height': 120, 'dt': DataType.RGB888},
        {'vc': 3, 'width': 80, 'height': 60, 'dt': DataType.YUV422_8BIT}
    ]

    # Start concurrent transmissions
    for i, vc_config in enumerate(vc_configs):
        task = cocotb.start_soon(
            tx_model.send_frame(
                vc_config['width'], vc_config['height'],
                vc_config['dt'], vc_config['vc'], i
            )
        )
        frame_tasks.append(task)

    # Wait for all transmissions to complete
    await asyncio.gather(*frame_tasks)

    # Verify all frames received
    received_frames = {}
    for vc_config in vc_configs:
        vc = vc_config['vc']
        frame_received = await rx_model.wait_for_frame(vc, timeout_ns=100000)
        assert frame_received, f"Frame not received on VC{vc}"

        frame_data = rx_model.get_frame_data(vc)
        assert frame_data is not None, f"No frame data for VC{vc}"
        received_frames[vc] = len(frame_data)

        cocotb.log.info(f"VC{vc}: {len(frame_data)} bytes received")

    # Verify correct frame sizes
    for vc_config in vc_configs:
        vc = vc_config['vc']
        expected_size = vc_config['width'] * vc_config['height']
        if vc_config['dt'] == DataType.RGB888:
            expected_size *= 3
        elif vc_config['dt'] == DataType.YUV422_8BIT:
            expected_size *= 2
        elif vc_config['dt'] == DataType.RAW10:
            expected_size = int(expected_size * 1.25)

        actual_size = received_frames[vc]
        assert actual_size == expected_size, \
            f"VC{vc} size mismatch: expected {expected_size}, got {actual_size}"

    cocotb.log.info("Multi-lane virtual channel test completed")


@cocotb.test()
async def test_multilane_error_resilience(dut):
    """Test error resilience in multi-lane configuration"""

    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Configure multi-lane with error injection
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=1000,
        continuous_clock=True,
        lane_distribution_enabled=True,
        inject_ecc_errors=True,
        inject_checksum_errors=True,
        error_injection_rate=0.1
    )

    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)

    # Enable error injection
    tx_model.enable_error_injection(0.1)

    # Track errors
    error_log = []

    async def error_handler(error_type, packet, message):
        error_log.append((error_type, cocotb.utils.get_sim_time('ns')))
        cocotb.log.info(f"Error detected: {error_type} at {cocotb.utils.get_sim_time('ns')}ns")

    rx_model.on_error_detected = error_handler

    # Send multiple frames with errors
    frames_sent = 0
    frames_received = 0

    for frame_num in range(10):
        await tx_model.send_frame(160, 120, DataType.RAW8, 0, frame_num)
        frames_sent += 1

        # Check if frame was received (some may be lost due to errors)
        frame_received = await rx_model.wait_for_frame(0, timeout_ns=50000)
        if frame_received:
            frames_received += 1

        await Timer(5000, units="ns")

    # Check error statistics
    tx_stats = tx_model.get_statistics()
    rx_stats = rx_model.get_statistics()

    assert tx_stats['errors_injected'] > 0, "No errors were injected"
    assert len(error_log) > 0, "No errors were detected"

    # Verify system resilience - should receive most frames despite errors
    success_rate = frames_received / frames_sent
    assert success_rate > 0.5, f"Poor error resilience: {success_rate:.2f} success rate"

    cocotb.log.info(f"Error resilience test: {frames_received}/{frames_sent} frames received, "
                   f"{len(error_log)} errors detected, {success_rate:.2f} success rate")


@cocotb.test()
async def test_lane_skew_tolerance(dut):
    """Test tolerance to lane skew in multi-lane configuration"""

    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Configure multi-lane D-PHY
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=1500,
        continuous_clock=True,
        lane_distribution_enabled=True
    )

    # Create bus and models
    bus = Csi2DPhyBus(dut, lane_count=4)
    tx_model = Csi2TxModel(bus, config)
    rx_model = Csi2RxModel(bus, config)

    # Enable strict timing validation to detect skew
    rx_model.enable_strict_timing_validation(True)

    # Test with high-speed data transmission
    for test_round in range(3):
        cocotb.log.info(f"Lane skew test round {test_round + 1}")

        # Send high-rate data stream
        for packet_num in range(20):
            # Alternate between different packet types
            if packet_num % 4 == 0:
                packet = Csi2ShortPacket.frame_start(0, packet_num // 4)
            elif packet_num % 4 == 1:
                test_data = bytes([(packet_num + i) & 0xFF for i in range(64)])
                packet = Csi2LongPacket(0, DataType.RAW8, test_data)
            elif packet_num % 4 == 2:
                packet = Csi2ShortPacket.line_start(0, packet_num)
            else:
                packet = Csi2ShortPacket.line_end(0, packet_num)

            await tx_model.send_packet(packet)
            await Timer(500, units="ns")  # Rapid transmission

        await Timer(10000, units="ns")

    # Check reception statistics
    stats = rx_model.get_statistics()

    # Should receive packets despite potential lane skew
    assert stats['packets_received'] > 30, \
        f"Too few packets received: {stats['packets_received']}"

    # Timing violations indicate lane skew issues (expected in rapid transmission)
    if stats['timing_violations'] > 0:
        cocotb.log.info(f"Detected {stats['timing_violations']} timing violations (lane skew)")

    cocotb.log.info(f"Lane skew tolerance test: {stats['packets_received']} packets received, "
                   f"{stats['timing_violations']} timing violations")


@cocotb.test()
async def test_cphy_trio_scaling(dut):
    """Test C-PHY trio scaling performance"""

    # Create clock
    clock = Clock(dut.clk, 8, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(200, units="ns")
    dut.reset_n.value = 1
    await Timer(200, units="ns")

    # Test different trio configurations
    trio_configs = [1, 2, 3]
    performance_results = {}

    for trios in trio_configs:
        cocotb.log.info(f"Testing {trios}-trio C-PHY configuration")

        # Configure C-PHY
        config = Csi2Config(
            phy_type=PhyType.CPHY,
            trio_count=trios,
            bit_rate_mbps=1500,
            lane_distribution_enabled=True
        )

        # Create bus and models
        bus = Csi2CPhyBus(dut, trio_count=trios)
        tx_model = Csi2TxModel(bus, config)
        rx_model = Csi2RxModel(bus, config)

        # Test frame transmission
        width, height = 320, 240
        start_time = cocotb.utils.get_sim_time('ns')

        await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
        frame_received = await rx_model.wait_for_frame(0, timeout_ns=200000)

        end_time = cocotb.utils.get_sim_time('ns')

        assert frame_received, f"{trios}-trio frame not received"

        # Calculate performance
        transmission_time = end_time - start_time
        frame_size = width * height
        throughput_mbps = (frame_size * 8) / (transmission_time / 1000)

        performance_results[trios] = {
            'time_ns': transmission_time,
            'throughput_mbps': throughput_mbps
        }

        cocotb.log.info(f"{trios}-trio: {transmission_time}ns, {throughput_mbps:.1f} Mbps")

        # Reset for next test
        await tx_model.reset()
        await rx_model.reset()
        await Timer(2000, units="ns")

    # Verify trio scaling
    if len(performance_results) > 1:
        single_trio = performance_results[1]['throughput_mbps']

        for trios in performance_results:
            if trios > 1:
                throughput = performance_results[trios]['throughput_mbps']
                scaling_factor = throughput / single_trio

                cocotb.log.info(f"C-PHY scaling 1 to {trios} trios: {scaling_factor:.2f}x")

    cocotb.log.info("C-PHY trio scaling test completed")


@cocotb.test()
async def test_mixed_phy_comparison(dut):
    """Compare D-PHY and C-PHY with equivalent configurations"""

    # Create clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Test configurations with similar bandwidth
    test_configs = [
        {
            'name': 'D-PHY 4-lane 1Gbps',
            'phy_type': PhyType.DPHY,
            'lane_count': 4,
            'trio_count': 0,
            'bit_rate': 1000
        },
        {
            'name': 'C-PHY 1-trio 1.5Gbps',
            'phy_type': PhyType.CPHY,
            'lane_count': 0,
            'trio_count': 1,
            'bit_rate': 1500
        }
    ]

    results = {}

    for test_config in test_configs:
        cocotb.log.info(f"Testing {test_config['name']}")

        # Create configuration
        if test_config['phy_type'] == PhyType.DPHY:
            config = Csi2Config(
                phy_type=PhyType.DPHY,
                lane_count=test_config['lane_count'],
                bit_rate_mbps=test_config['bit_rate'],
                continuous_clock=True,
                lane_distribution_enabled=True
            )
            bus = Csi2DPhyBus(dut, lane_count=test_config['lane_count'])
        else:
            config = Csi2Config(
                phy_type=PhyType.CPHY,
                trio_count=test_config['trio_count'],
                bit_rate_mbps=test_config['bit_rate'],
                lane_distribution_enabled=True
            )
            bus = Csi2CPhyBus(dut, trio_count=test_config['trio_count'])

        # Create models
        tx_model = Csi2TxModel(bus, config)
        rx_model = Csi2RxModel(bus, config)

        # Test identical frame
        width, height = 640, 480
        start_time = cocotb.utils.get_sim_time('ns')

        await tx_model.send_frame(width, height, DataType.RAW8, 0, 0)
        frame_received = await rx_model.wait_for_frame(0, timeout_ns=300000)

        end_time = cocotb.utils.get_sim_time('ns')

        assert frame_received, f"{test_config['name']} frame not received"

        # Calculate metrics
        transmission_time = end_time - start_time
        frame_size = width * height
        throughput_mbps = (frame_size * 8) / (transmission_time / 1000)

        results[test_config['name']] = {
            'time_ns': transmission_time,
            'throughput_mbps': throughput_mbps,
            'efficiency': throughput_mbps / test_config['bit_rate']
        }

        cocotb.log.info(f"{test_config['name']}: {transmission_time}ns, "
                       f"{throughput_mbps:.1f} Mbps, "
                       f"{results[test_config['name']]['efficiency']:.2f} efficiency")

        # Cleanup
        await tx_model.reset()
        await rx_model.reset()
        await Timer(2000, units="ns")

    # Compare results
    config_names = list(results.keys())
    if len(config_names) == 2:
        dphy_result = results[config_names[0]]
        cphy_result = results[config_names[1]]

        time_ratio = dphy_result['time_ns'] / cphy_result['time_ns']
        throughput_ratio = dphy_result['throughput_mbps'] / cphy_result['throughput_mbps']

        cocotb.log.info(f"Comparison: D-PHY vs C-PHY")
        cocotb.log.info(f"  Time ratio: {time_ratio:.2f}")
        cocotb.log.info(f"  Throughput ratio: {throughput_ratio:.2f}")

    cocotb.log.info("Mixed PHY comparison test completed")


@cocotb.test()
async def test_high_throughput_streaming(dut):
    """Test high throughput continuous streaming"""

    # Create high-speed clock
    clock = Clock(dut.clk, 5, units="ns")  # 200 MHz
    cocotb.start_soon(clock.start())

    # Reset
    dut.reset_n.value = 0
    await Timer(100, units="ns")
    dut.reset_n.value = 1
    await Timer(100, units="ns")

    # Configure for maximum throughput
    config = Csi2Config(
        phy_type=PhyType.DPHY,
        lane_count=4,
        bit_rate_mbps=2500,  # High speed
        continuous_clock=True,
        lane_distribution_enabled=True
    )

    # Create models
    bus = Csi2DPhyBus(dut, lane_count=4)
    image_tx = Csi2ImageTransmitter(bus, config)
    image_rx = Csi2ImageReceiver(bus, config)

    # Start continuous video stream
    image_tx.start_video_stream(
        width=1280, height=720,
        data_type=DataType.RAW8,
        frame_rate_fps=60.0
    )

    # Monitor reception for a period
    monitor_time_ns = 100000  # 100 microseconds
    start_time = cocotb.utils.get_sim_time('ns')
    frames_received = 0

    while cocotb.utils.get_sim_time('ns') - start_time < monitor_time_ns:
        frame_received = await image_rx.wait_for_frame(0, timeout_ns=10000)
        if frame_received:
            frames_received += 1
            # Validate frame format
            is_valid = await image_rx.validate_frame_format(0, 1280, 720, DataType.RAW8)
            assert is_valid, f"Invalid frame format detected"

        await Timer(1000, units="ns")

    # Stop streaming
    image_tx.stop_continuous_transmission()

    # Calculate achieved frame rate
    actual_time_s = monitor_time_ns / 1e9
    achieved_fps = frames_received / actual_time_s

    # Check statistics
    tx_stats = image_tx.get_statistics()
    rx_stats = image_rx.get_statistics()

    cocotb.log.info(f"High throughput streaming test:")
    cocotb.log.info(f"  Frames received: {frames_received}")
    cocotb.log.info(f"  Achieved FPS: {achieved_fps:.1f}")
    cocotb.log.info(f"  TX stats: {tx_stats['frames_sent']} frames sent")
    cocotb.log.info(f"  RX stats: {rx_stats['frames_received']} frames received")

    # Verify reasonable performance
    assert frames_received > 0, "No frames received during streaming"
    assert rx_stats['packets_received'] > 0, "No packets received"

    cocotb.log.info("High throughput streaming test completed")
