/*
 * C-PHY specific CSI-2 test module
 *
 * Copyright (c) 2024 CSI-2 Extension Contributors
 */

`timescale 1ns / 1ps

module test_csi2_cphy (
    input  wire clk,
    input  wire reset_n,
    
    // C-PHY Trio 0 (A, B, C signals)
    output wire trio0_a,
    output wire trio0_b,
    output wire trio0_c,
    
    // C-PHY Trio 1 (for multi-trio tests)
    output wire trio1_a,
    output wire trio1_b,
    output wire trio1_c,
    
    // C-PHY Trio 2 (for maximum trio tests)
    output wire trio2_a,
    output wire trio2_b,
    output wire trio2_c,
    
    // Control and status
    input  wire enable,
    output reg  hs_active,
    output reg  lp_active,
    output reg [3:0] trio_state,
    
    // Received data interface
    output reg [7:0] rx_data,
    output reg       rx_valid,
    output reg [1:0] rx_vc,
    output reg [5:0] rx_dt
);

    // C-PHY 3-Phase States
    localparam STATE_0 = 3'b001;  // ABC = 001
    localparam STATE_1 = 3'b010;  // ABC = 010
    localparam STATE_2 = 3'b100;  // ABC = 100
    localparam STATE_3 = 3'b110;  // ABC = 110
    localparam STATE_4 = 3'b101;  // ABC = 101
    localparam STATE_5 = 3'b011;  // ABC = 011
    localparam STATE_6 = 3'b000;  // ABC = 000
    
    // Low Power states
    localparam LP_000 = 3'b000;
    localparam LP_111 = 3'b111;
    
    // Receiver state machine
    localparam RX_IDLE = 4'h0;
    localparam RX_HS_PREP = 4'h1;
    localparam RX_HS_SYNC = 4'h2;
    localparam RX_HS_DATA = 4'h3;
    localparam RX_LP_EXIT = 4'h4;
    
    reg [3:0] rx_state;
    reg [3:0] rx_next_state;
    
    // Trio signal states
    wire [2:0] trio0_state = {trio0_c, trio0_b, trio0_a};
    wire [2:0] trio1_state = {trio1_c, trio1_b, trio1_a};
    wire [2:0] trio2_state = {trio2_c, trio2_b, trio2_a};
    
    // State transition detection
    reg [2:0] prev_trio0_state;
    reg [2:0] current_trio0_state;
    reg state_transition;
    
    // Symbol and byte recovery
    reg [2:0] symbol_buffer [0:2];  // Store 3 symbols per byte
    reg [1:0] symbol_count;
    reg [7:0] decoded_byte;
    reg byte_ready;
    
    // Packet parsing
    reg [7:0] header_bytes [0:3];
    reg [1:0] header_count;
    reg packet_active;
    reg [15:0] payload_count;
    reg [15:0] payload_length;
    
    // State transition tracking
    always @(posedge clk) begin
        if (!reset_n) begin
            prev_trio0_state <= LP_111;
            current_trio0_state <= LP_111;
            state_transition <= 1'b0;
        end else begin
            prev_trio0_state <= current_trio0_state;
            current_trio0_state <= trio0_state;
            state_transition <= (prev_trio0_state != current_trio0_state);
        end
    end
    
    // C-PHY symbol decoder (simplified)
    always @(posedge clk) begin
        if (!reset_n) begin
            symbol_count <= 2'h0;
            decoded_byte <= 8'h00;
            byte_ready <= 1'b0;
        end else if (hs_active && state_transition) begin
            // Store symbol transitions
            symbol_buffer[symbol_count] <= prev_trio0_state;
            symbol_count <= symbol_count + 1;
            
            if (symbol_count == 2) begin
                // 3 symbols collected, decode to byte
                decoded_byte <= decode_symbols(symbol_buffer[0], symbol_buffer[1], symbol_buffer[2]);
                byte_ready <= 1'b1;
                symbol_count <= 0;
            end else begin
                byte_ready <= 1'b0;
            end
        end else begin
            byte_ready <= 1'b0;
        end
    end
    
    // Simplified C-PHY symbol decoder function
    function [7:0] decode_symbols;
        input [2:0] sym0, sym1, sym2;
        begin
            // Simplified decoding - in reality this would be much more complex
            // This maps symbol transitions back to data bits
            decode_symbols = {sym2[1:0], sym1[1:0], sym0[1:0], sym0[2], sym1[2]};
        end
    endfunction
    
    // Main receiver state machine
    always @(posedge clk) begin
        if (!reset_n) begin
            rx_state <= RX_IDLE;
            hs_active <= 1'b0;
            lp_active <= 1'b1;
            trio_state <= 4'h0;
            rx_data <= 8'h00;
            rx_valid <= 1'b0;
            rx_vc <= 2'h0;
            rx_dt <= 6'h00;
            header_count <= 2'h0;
            packet_active <= 1'b0;
            payload_count <= 16'h0;
            payload_length <= 16'h0;
        end else begin
            rx_state <= rx_next_state;
            
            // Packet processing
            if (byte_ready) begin
                if (!packet_active) begin
                    // Header processing
                    header_bytes[header_count] <= decoded_byte;
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
                            process_short_packet(header_bytes[0][5:0], header_bytes[0][7:6], 
                                                {header_bytes[2], header_bytes[1]});
                        end
                    end
                end else begin
                    // Payload processing
                    rx_data <= decoded_byte;
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
                if (trio0_state == LP_000) begin
                    rx_next_state = RX_HS_PREP;
                end
            end
            
            RX_HS_PREP: begin
                if (trio0_state != LP_000 && trio0_state != LP_111) begin
                    rx_next_state = RX_HS_SYNC;
                end else if (trio0_state == LP_111) begin
                    rx_next_state = RX_IDLE;
                end
            end
            
            RX_HS_SYNC: begin
                rx_next_state = RX_HS_DATA;
            end
            
            RX_HS_DATA: begin
                if (trio0_state == LP_111) begin
                    rx_next_state = RX_LP_EXIT;
                end
            end
            
            RX_LP_EXIT: begin
                rx_next_state = RX_IDLE;
            end
        endcase
    end
    
    // Update status signals
    always @(posedge clk) begin
        if (!reset_n) begin
            hs_active <= 1'b0;
            lp_active <= 1'b1;
            trio_state <= 4'h0;
        end else begin
            case (rx_state)
                RX_IDLE: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b1;
                    trio_state <= 4'h1;
                end
                
                RX_HS_PREP: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b0;
                    trio_state <= 4'h2;
                end
                
                RX_HS_SYNC,
                RX_HS_DATA: begin
                    hs_active <= 1'b1;
                    lp_active <= 1'b0;
                    trio_state <= 4'h3;
                end
                
                RX_LP_EXIT: begin
                    hs_active <= 1'b0;
                    lp_active <= 1'b0;
                    trio_state <= 4'h4;
                end
            endcase
        end
    end
    
    // Process short packets
    task process_short_packet;
        input [5:0] data_type;
        input [1:0] virtual_channel;
        input [15:0] word_count;
        begin
            case (data_type)
                6'h00: begin // Frame Start
                    $display("C-PHY Frame Start: VC=%d, Frame=%d", virtual_channel, word_count);
                end
                
                6'h01: begin // Frame End
                    $display("C-PHY Frame End: VC=%d, Frame=%d", virtual_channel, word_count);
                end
                
                6'h02: begin // Line Start
                    $display("C-PHY Line Start: VC=%d, Line=%d", virtual_channel, word_count);
                end
                
                6'h03: begin // Line End
                    $display("C-PHY Line End: VC=%d, Line=%d", virtual_channel, word_count);
                end
                
                default: begin
                    $display("C-PHY Unknown short packet: DT=0x%02x, VC=%d, Data=0x%04x", 
                            data_type, virtual_channel, word_count);
                end
            endcase
        end
    endtask
    
    // Generate test trio signals (for loopback testing)
    reg [2:0] test_trio_counter;
    reg [2:0] test_trio_state;
    
    always @(posedge clk) begin
        if (!reset_n) begin
            test_trio_counter <= 3'h0;
            test_trio_state <= LP_111;
        end else if (enable) begin
            test_trio_counter <= test_trio_counter + 1;
            
            // Cycle through different 3-phase states
            case (test_trio_counter)
                3'h0: test_trio_state <= STATE_0;
                3'h1: test_trio_state <= STATE_1;
                3'h2: test_trio_state <= STATE_2;
                3'h3: test_trio_state <= STATE_3;
                3'h4: test_trio_state <= STATE_4;
                3'h5: test_trio_state <= STATE_5;
                3'h6: test_trio_state <= STATE_6;
                3'h7: test_trio_state <= LP_111;
            endcase
        end else begin
            test_trio_state <= LP_111;
        end
    end
    
    // Drive trio signals when enabled (for testing)
    assign trio0_a = enable ? test_trio_state[0] : 1'bz;
    assign trio0_b = enable ? test_trio_state[1] : 1'bz;
    assign trio0_c = enable ? test_trio_state[2] : 1'bz;
    
    // Initially tri-state other trios
    assign trio1_a = 1'bz;
    assign trio1_b = 1'bz;
    assign trio1_c = 1'bz;
    assign trio2_a = 1'bz;
    assign trio2_b = 1'bz;
    assign trio2_c = 1'bz;
    
    // Debug output for state transitions
    always @(posedge clk) begin
        if (state_transition && hs_active) begin
            $display("C-PHY State Transition: %b -> %b at time %t", 
                    prev_trio0_state, current_trio0_state, $time);
        end
        
        if (byte_ready) begin
            $display("C-PHY Decoded Byte: 0x%02x at time %t", decoded_byte, $time);
        end
    end
    
endmodule
