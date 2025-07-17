/*
 * Multi-lane CSI-2 test module
 *
 * Copyright (c) 2024 CSI-2 Extension Contributors
 */

`timescale 1ns / 1ps

module test_csi2_multilane (
    input  wire clk,
    input  wire reset_n,
    
    // D-PHY Clock Lane
    output wire clk_p,
    output wire clk_n,
    
    // D-PHY Data Lanes (8 lanes maximum)
    output wire [7:0] data_p,
    output wire [7:0] data_n,
    
    // C-PHY Trios (3 trios maximum)
    output wire [2:0] trio_a,
    output wire [2:0] trio_b,
    output wire [2:0] trio_c,
    
    // Control and configuration
    input  wire [3:0] active_lanes,
    input  wire [1:0] active_trios,
    input  wire       phy_mode,      // 0=D-PHY, 1=C-PHY
    input  wire       enable,
    
    // Status and monitoring
    output reg  [7:0] lane_status,
    output reg  [2:0] trio_status,
    output reg        deskew_locked,
    output reg        frame_sync,
    output reg [15:0] throughput_mbps,
    
    // Received data interface
    output reg [7:0]  rx_data,
    output reg        rx_valid,
    output reg [1:0]  rx_vc,
    output reg [5:0]  rx_dt,
    output reg [3:0]  rx_lane_id
);

    // Lane deskew parameters
    parameter DESKEW_WINDOW = 16;  // Deskew window in UI
    parameter MAX_SKEW = 8;        // Maximum allowed skew
    
    // Multi-lane receiver state machine
    localparam ML_IDLE = 4'h0;
    localparam ML_SYNC = 4'h1;
    localparam ML_DESKEW = 4'h2;
    localparam ML_DATA = 4'h3;
    localparam ML_ERROR = 4'h4;
    
    reg [3:0] ml_state;
    reg [3:0] ml_next_state;
    
    // Per-lane D-PHY status
    wire [7:0] dphy_hs_active;
    wire [7:0] dphy_lp_active;
    reg  [7:0] dphy_byte_ready;
    reg  [7:0] dphy_bytes [0:7];
    
    // Detect HS mode on each lane
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane_detection
            wire lane_p = data_p[i];
            wire lane_n = data_n[i];
            wire [1:0] lane_state = {lane_p, lane_n};
            
            assign dphy_hs_active[i] = (lane_state == 2'b01) || (lane_state == 2'b10);
            assign dphy_lp_active[i] = (lane_state == 2'b11) || (lane_state == 2'b00);
        end
    endgenerate
    
    // Clock recovery and data sampling (simplified)
    reg [7:0] bit_counters [0:7];
    reg [7:0] byte_shifters [0:7];
    
    always @(posedge clk) begin
        if (!reset_n) begin
            for (integer j = 0; j < 8; j = j + 1) begin
                bit_counters[j] <= 8'h0;
                byte_shifters[j] <= 8'h0;
                dphy_byte_ready[j] <= 1'b0;
                dphy_bytes[j] <= 8'h00;
            end
        end else begin
            for (integer j = 0; j < 8; j = j + 1) begin
                if (dphy_hs_active[j] && (j < active_lanes)) begin
                    // Sample data bit
                    byte_shifters[j] <= {data_p[j], byte_shifters[j][7:1]};
                    bit_counters[j] <= bit_counters[j] + 1;
                    
                    if (bit_counters[j] == 7) begin
                        dphy_bytes[j] <= {data_p[j], byte_shifters[j][7:1]};
                        dphy_byte_ready[j] <= 1'b1;
                        bit_counters[j] <= 0;
                    end else begin
                        dphy_byte_ready[j] <= 1'b0;
                    end
                end else begin
                    dphy_byte_ready[j] <= 1'b0;
                    bit_counters[j] <= 0;
                end
            end
        end
    end
    
    // Lane deskew logic
    reg [7:0] deskew_counters [0:7];
    reg [7:0] deskew_targets [0:7];
    reg [7:0] lanes_synced;
    
    always @(posedge clk) begin
        if (!reset_n) begin
            for (integer k = 0; k < 8; k = k + 1) begin
                deskew_counters[k] <= 8'h0;
                deskew_targets[k] <= 8'h0;
            end
            lanes_synced <= 8'h00;
            deskew_locked <= 1'b0;
        end else if (ml_state == ML_DESKEW) begin
            // Simple deskew algorithm
            for (integer k = 0; k < 8; k = k + 1) begin
                if (k < active_lanes) begin
                    if (dphy_byte_ready[k]) begin
                        deskew_counters[k] <= deskew_counters[k] + 1;
                        
                        // Check for sync pattern (simplified)
                        if (dphy_bytes[k] == 8'hB8) begin // SoT sequence
                            deskew_targets[k] <= deskew_counters[k];
                            lanes_synced[k] <= 1'b1;
                        end
                    end
                end else begin
                    lanes_synced[k] <= 1'b1; // Unused lanes are "synced"
                end
            end
            
            // Check if all active lanes are synced
            deskew_locked <= (lanes_synced[active_lanes-1:0] == {active_lanes{1'b1}});
        end
    end
    
    // Multi-lane data merging
    reg [7:0] merged_data;
    reg       merged_valid;
    reg [2:0] merge_lane_select;
    reg [7:0] merge_byte_count;
    
    always @(posedge clk) begin
        if (!reset_n) begin
            merged_data <= 8'h00;
            merged_valid <= 1'b0;
            merge_lane_select <= 3'h0;
            merge_byte_count <= 8'h0;
        end else if (ml_state == ML_DATA && deskew_locked) begin
            // Round-robin merge from active lanes
            if (dphy_byte_ready[merge_lane_select] && (merge_lane_select < active_lanes)) begin
                merged_data <= dphy_bytes[merge_lane_select];
                merged_valid <= 1'b1;
                merge_byte_count <= merge_byte_count + 1;
                
                // Next lane
                if (merge_lane_select >= active_lanes - 1) begin
                    merge_lane_select <= 0;
                end else begin
                    merge_lane_select <= merge_lane_select + 1;
                end
            end else begin
                merged_valid <= 1'b0;
                
                // Advance to next lane if current lane not ready
                if (merge_lane_select >= active_lanes - 1) begin
                    merge_lane_select <= 0;
                end else begin
                    merge_lane_select <= merge_lane_select + 1;
                end
            end
        end else begin
            merged_valid <= 1'b0;
        end
    end
    
    // Packet parsing for merged data
    reg [7:0] packet_header [0:3];
    reg [1:0] header_byte_count;
    reg       packet_active;
    reg [15:0] payload_remaining;
    
    always @(posedge clk) begin
        if (!reset_n) begin
            header_byte_count <= 2'h0;
            packet_active <= 1'b0;
            payload_remaining <= 16'h0;
            rx_data <= 8'h00;
            rx_valid <= 1'b0;
            rx_vc <= 2'h0;
            rx_dt <= 6'h00;
            rx_lane_id <= 4'h0;
        end else if (merged_valid) begin
            if (!packet_active) begin
                // Collecting header
                packet_header[header_byte_count] <= merged_data;
                header_byte_count <= header_byte_count + 1;
                
                if (header_byte_count == 3) begin
                    // Header complete
                    packet_active <= 1'b1;
                    header_byte_count <= 0;
                    payload_remaining <= {packet_header[2], packet_header[1]};
                    
                    // Extract packet info
                    rx_vc <= packet_header[0][7:6];
                    rx_dt <= packet_header[0][5:0];
                    rx_lane_id <= merge_lane_select;
                    
                    if ({packet_header[2], packet_header[1]} == 0) begin
                        // Short packet
                        packet_active <= 1'b0;
                        process_short_packet(packet_header[0][5:0], packet_header[0][7:6], 
                                            {packet_header[2], packet_header[1]});
                    end
                end
            end else begin
                // Payload data
                rx_data <= merged_data;
                rx_valid <= 1'b1;
                rx_lane_id <= merge_lane_select;
                payload_remaining <= payload_remaining - 1;
                
                if (payload_remaining <= 1) begin
                    packet_active <= 1'b0;
                    rx_valid <= 1'b0;
                end
            end
        end else begin
            rx_valid <= 1'b0;
        end
    end
    
    // Multi-lane state machine
    always @(posedge clk) begin
        if (!reset_n) begin
            ml_state <= ML_IDLE;
        end else begin
            ml_state <= ml_next_state;
        end
    end
    
    always @(*) begin
        ml_next_state = ml_state;
        
        case (ml_state)
            ML_IDLE: begin
                if (enable && phy_mode == 0) begin // D-PHY mode
                    if (|dphy_hs_active[active_lanes-1:0]) begin
                        ml_next_state = ML_SYNC;
                    end
                end
            end
            
            ML_SYNC: begin
                if (&dphy_hs_active[active_lanes-1:0]) begin
                    ml_next_state = ML_DESKEW;
                end else if (!(|dphy_hs_active[active_lanes-1:0])) begin
                    ml_next_state = ML_IDLE;
                end
            end
            
            ML_DESKEW: begin
                if (deskew_locked) begin
                    ml_next_state = ML_DATA;
                end else if (!(|dphy_hs_active[active_lanes-1:0])) begin
                    ml_next_state = ML_IDLE;
                end
            end
            
            ML_DATA: begin
                if (!deskew_locked || !(|dphy_hs_active[active_lanes-1:0])) begin
                    ml_next_state = ML_IDLE;
                end
            end
        endcase
    end
    
    // Status reporting
    always @(posedge clk) begin
        if (!reset_n) begin
            lane_status <= 8'h00;
            trio_status <= 3'h0;
            frame_sync <= 1'b0;
            throughput_mbps <= 16'h0000;
        end else begin
            // Lane status
            lane_status <= dphy_hs_active;
            
            // Frame sync detection (simplified)
            frame_sync <= packet_active && (rx_dt == 6'h00); // Frame start
            
            // Throughput estimation (simplified)
            if (ml_state == ML_DATA) begin
                throughput_mbps <= active_lanes * 100; // Rough estimate
            end else begin
                throughput_mbps <= 16'h0000;
            end
        end
    end
    
    // Process short packets
    task process_short_packet;
        input [5:0] data_type;
        input [1:0] virtual_channel;
        input [15:0] word_count;
        begin
            case (data_type)
                6'h00: $display("Multi-lane Frame Start: VC=%d, Frame=%d", virtual_channel, word_count);
                6'h01: $display("Multi-lane Frame End: VC=%d, Frame=%d", virtual_channel, word_count);
                6'h02: $display("Multi-lane Line Start: VC=%d, Line=%d", virtual_channel, word_count);
                6'h03: $display("Multi-lane Line End: VC=%d, Line=%d", virtual_channel, word_count);
            endcase
        end
    endtask
    
    // C-PHY trio handling (simplified placeholder)
    reg [2:0] cphy_test_state;
    
    always @(posedge clk) begin
        if (!reset_n) begin
            cphy_test_state <= 3'b111;
            trio_status <= 3'h0;
        end else if (phy_mode == 1 && enable) begin
            // Simple C-PHY state cycling for testing
            cphy_test_state <= cphy_test_state + 1;
            trio_status <= active_trios;
        end else begin
            cphy_test_state <= 3'b111;
            trio_status <= 3'h0;
        end
    end
    
    // Drive signals
    assign clk_p = enable ? clk : 1'bz;
    assign clk_n = enable ? ~clk : 1'bz;
    
    // D-PHY data lanes (initially tri-state)
    assign data_p = 8'hzz;
    assign data_n = 8'hzz;
    
    // C-PHY trios
    assign trio_a = phy_mode ? {3{cphy_test_state[0]}} : 3'bzz;
    assign trio_b = phy_mode ? {3{cphy_test_state[1]}} : 3'bzz;
    assign trio_c = phy_mode ? {3{cphy_test_state[2]}} : 3'bzz;
    
    // Debug output
    always @(posedge clk) begin
        if (deskew_locked && merged_valid && packet_active) begin
            $display("Multi-lane RX: Lane=%d, VC=%d, DT=0x%02x, Data=0x%02x at %t", 
                    rx_lane_id, rx_vc, rx_dt, merged_data, $time);
        end
    end
    
endmodule
