/*
 * Basic CSI-2 test module - Interface only
 *
 * Copyright (c) 2024 CSI-2 Extension Contributors
 */

`timescale 1ns / 1ps

module test_csi2_basic (

    // D-PHY Clock Lane
    inout wire clk_p,
    inout wire clk_n,

    // D-PHY Data Lane 0
    inout wire data0_p,
    inout wire data0_n,

    // D-PHY Data Lane 1
    inout wire data1_p,
    inout wire data1_n,

    // D-PHY Data Lane 2
    inout wire data2_p,
    inout wire data2_n,

    // D-PHY Data Lane 3
    inout wire data3_p,
    inout wire data3_n
);



endmodule
