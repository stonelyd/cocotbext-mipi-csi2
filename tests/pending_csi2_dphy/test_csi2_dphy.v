/*
 * D-PHY specific CSI-2 test module
 *
 * Copyright (c) 2024 CSI-2 Extension Contributors
 */

`timescale 1ns / 1ps

module test_csi2_dphy (
    input  wire clk,
    input  wire reset_n,
    
    // D-PHY Clock Lane
    output wire clk_p,
    output wire clk_n,
    
    // D-PHY Data Lanes (up to 4 lanes)
    output wire data0_p,
    output wire data0_n,
    output wire data1_p,
    output wire data1_n,
    output wire data2_p,
    output wire data2_n,
    output wire data3_p,
    output wire data3_n,
    
    // Control and status
    input  wire enable,
    output reg  hs_active,
    output reg  lp_active,
    output reg [3:0] lane_state,
    
    // Received data interface
    output reg [7:0] rx_data,
    output reg       rx_valid,
    output reg [1:0] rx_vc,
    output reg [5:0] rx_dt
);

    // D-PHY Lane States
    localparam LP_00 = 2'b00;
    localparam LP_01 = 2'b01;
    localparam LP_10 = 2'b10;
    localparam LP_11 = 2'b11;
    
    // Receiver state machine
    localparam RX_IDLE = 4'h0;
    localparam RX_HS_PREP = 4'h1;
    localparam RX_HS_SYNC = 4'h2;
    localparam RX_HS_DATA = 4'h3;
    localparam RX_HS_TRAIL = 4'h4;
    
    reg [3:0] rx_state;
    reg [3:0] rx_next_state;
    
    // Lane status detection
    wire [1:0] clk_lane_state = {clk_p, clk_n};
    wire [1:0] data0_lane_state = {data0_p, data0_n};
    wire [1:0] data1_lane_state = {data1_p, data1_n};
    wire [1:0] data2_lane_state = {data2_p, data2_n};
    wire [1:0] data3_lane_state = {data3_p, data3_n};
    
    // HS detection
    wire hs_clock_detected = (clk_lane_state == LP_01) || (clk_lane_state == LP_10);
    wire hs_data0_detected = (data0_lane_state == LP_01) || (data0_lane_state == LP_10);
    wire hs_data1_detected = (data1_lane_state == LP_01) || (data1_lane_state == LP_10);
    wire hs_data2_detected = (data2_lane_state == LP_01) || (data2_lane_state == LP_10);
    wire hs_data3_detected = (data3_lane_state == LP_01) || (data3_lane_state == LP_10);
    
    // Multi-lane HS detection
    wire hs_detected = hs_data0_detected; // Simplified - use data lane 0
    
    // Byte recovery from differential signals
    reg [7:0] byte_shifter;
    reg [2:0] bit_counter;
    reg byte_ready;
    
    // Packet parsing
    reg [7:0] header_bytes [0:3];
    reg [1:0] header_count;
    reg packet_active;
    reg [15:0] payload_count;
    reg [15:0] payload_length;
    
    // State machine for D-PHY reception
    always @(posedge clk) begin
        if (!reset_n) begin
            rx_state <= RX_IDLE;
            hs_active <= 1'b0;
            lp_active <= 1'b1;
            lane_state <= 4'h0;
            rx_data <= 8'h00;
            rx_valid <= 1'b0;
            rx_vc <= 2'h0;
            rx_dt <= 6'h00;
            byte_shifter <= 8'h00;
            bit_counter <= 3'h0;
            byte_ready <= 1'b0;
            header_count <= 2'h0;
            packet_active <= 1'b0;
            payload_count <= 16'h0;
            payload_length <= 16'h0;
        end else begin
            rx_state <= rx_next_state;
            
            // Byte recovery
            if (hs_active) begin
                byte_shifter <= {data0_p, byte_shifter[7:1]};
                bit_counter <= bit_counter + 1;
                
                if (bit_counter == 7) begin
                    byte_ready <= 1'b1;
                    bit_counter <= 0;
                end else begin
                    byte_ready <= 1'b0;
                end
            end else begin
                byte_ready <= 1'b0;
                bit_counter <= 0;
            end
            
            // Packet processing
            if (byte_ready) begin
                if (!packet_active) begin
                    // Header processing
                    header_bytes[header_count] <= {data0_p, byte_shifter[7:1]};
                    header_count <= header_count + 1;
                    
                    if (header_count == 3) begin
                        // Header complete
                        packet_active <= 1'b1;
                        header_count <= 0;
                        payload_length <= {header_bytes[2], header_bytes[1]};
                        payload_count <= 0;
                        
                        // Extract VC and DT
                        rx_vc <= header_bytes[0][7:6];
                        rx_dt <= header_bytes[0][5:0];
                        
                        if ({header_bytes[2], header_bytes[1]} == 0) begin
                            // Short packet - complete
                            packet_active <= 1'b0;
                        end
                    end
                end else begin
                    // Payload processing
                    rx_data <= {data0_p, byte_shifter[7:1]};
                    rx_valid <= 1'b1;
                    payload_count <= payload_count + 1;
                    
                    if (payload_count >= payload_length + 1) begin // +2 for checksum, -1 for indexing
                        packet_active <= 1'b0;
                        rx_valid <= 1'b0;
                    end
                end
            end else begin
                rx_valid <= 1'b0;
            end
        end
    end
    
    // State machine logic
    always @(*) begin
        rx_next_state = rx_state;
        
        case (rx_state)
            RX_IDLE: begin
                if (data0_lane_state == LP_00) begin
                    rx_next_state = RX_HS_PREP;
                end
            end
            
            RX_HS_PREP: begin
                if (hs_detected) begin
                    rx_next_state = RX_HS_SYNC;
                end else if (data0_lane_state == LP_11) begin
                    rx_next_state = RX_IDLE;
                end
            end
            
            RX_HS_SYNC: begin
                rx_next_state = RX_HS_DATA;
            end
            
            RX_HS_DATA: begin
                if (!hs_detected && data0_lane_state == LP_11) begin
                    rx_next_state = RX_HS_TRAIL;
                end
            end
            
            RX_HS_TRAIL: begin
                rx_next_state = RX_IDLE;
            end
        endcase
    end
    
    // Update status signals
    always @(posedge clk) begin
        if (!reset_n) begin
            hs_active <= 1'b0;
            lp_active <= 1'b1;
            lane_state <= 4'h0;
        end else begin
            case (rx_state)
                RX_IDLE: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b1;
                    lane_state <= 4'h1;
                end
                
                RX_HS_PREP: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b0;
                    lane_state <= 4'h2;
                end
                
                RX_HS_SYNC,
                RX_HS_DATA: begin
                    hs_active <= 1'b1;
                    lp_active <= 1'b0;
                    lane_state <= 4'h3;
                end
                
                RX_HS_TRAIL: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b0;
                    lane_state <= 4'h4;
                end
            endcase
        end
    end
    
    // Generate differential clock (for loopback testing)
    reg clk_div2;
    always @(posedge clk) begin
        if (!reset_n) begin
            clk_div2 <= 1'b0;
        end else if (enable) begin
            clk_div2 <= ~clk_div2;
        end
    end
    
    // Drive clock lanes when enabled
    assign clk_p = enable ? clk_div2 : 1'bz;
    assign clk_n = enable ? ~clk_div2 : 1'bz;
    
    // Initially tri-state all data lanes
    assign data0_p = 1'bz;
    assign data0_n = 1'bz;
    assign data1_p = 1'bz;
    assign data1_n = 1'bz;
    assign data2_p = 1'bz;
    assign data2_n = 1'bz;
    assign data3_p = 1'bz;
    assign data3_n = 1'bz;
    
    // Debug output
    always @(posedge clk) begin
        if (byte_ready && packet_active) begin
            $display("D-PHY RX: VC=%d, DT=0x%02x, Data=0x%02x", 
                    rx_vc, rx_dt, {data0_p, byte_shifter[7:1]});
        end
    end
    
endmodule
