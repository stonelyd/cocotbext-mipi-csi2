/*
 * Basic CSI-2 test module
 *
 * Copyright (c) 2024 CSI-2 Extension Contributors
 */

`timescale 1ns / 1ps

module test_csi2_basic (
    input  wire clk,
    input  wire reset_n,

    // D-PHY Clock Lane
    output wire clk_p,
    output wire clk_n,

    // D-PHY Data Lane 0
    output wire data0_p,
    output wire data0_n,

    // D-PHY Data Lane 1 (for multi-lane tests)
    output wire data1_p,
    output wire data1_n,

    // Status signals
    output wire frame_valid,
    output reg  line_valid,
    output reg [7:0] pixel_data,
    output reg  pixel_valid
);

    // Simple CSI-2 receiver implementation for testing
    // This is a minimal implementation to validate the TX model

    reg [3:0] state;
    reg [7:0] byte_buffer;
    reg [2:0] bit_count;
    reg [15:0] packet_length;
    reg [15:0] byte_count;
    reg [15:0] timeout_counter;

    // State machine states
    localparam IDLE = 4'h0;
    localparam HS_SYNC = 4'h1;
    localparam HEADER = 4'h2;
    localparam PAYLOAD = 4'h3;
    localparam CHECKSUM = 4'h4;
    localparam SHORT_PACKET_PROCESS = 4'h5;

        // Decode differential signals (correct differential decoding)
    wire clk_lane = clk_p;
    // HS-0: p=0, n=1; HS-1: p=1, n=0
    // Only decode when signals are in valid HS state (complementary)
    wire data0_lane = (data0_p === 1'b1 && data0_n === 1'b0) ? 1'b1 :
                     (data0_p === 1'b0 && data0_n === 1'b1) ? 1'b0 :
                     1'b0; // Default to 0 for invalid states
    wire data1_lane = (data1_p === 1'b1 && data1_n === 1'b0) ? 1'b1 :
                     (data1_p === 1'b0 && data1_n === 1'b1) ? 1'b0 : 1'b0;

    // HS clock detection - detect when clock lane is active (complementary signals)
    wire hs_clock_active = (clk_p === 1'b1 && clk_n === 1'b0) || (clk_p === 1'b0 && clk_n === 1'b1);

    // HS data detection - detect when data lane is in HS mode (complementary signals)
    wire hs_data_active = (data0_p === 1'b1 && data0_n === 1'b0) || (data0_p === 1'b0 && data0_n === 1'b1);

    // Debug: monitor differential signals (reduced frequency)
    reg [15:0] debug_counter;
    always @(posedge clk) begin
        if (reset_n) begin
            debug_counter <= debug_counter + 1;
            if (debug_counter == 0 && (data0_p !== 1'bz && data0_n !== 1'bz)) begin
                $display("DUT: data0_p=%b, data0_n=%b, data0_lane=%b, clk_p=%b, clk_n=%b, hs_clock=%b",
                        data0_p, data0_n, data0_lane, clk_p, clk_n, hs_clock_active);
            end
        end else begin
            debug_counter <= 0;
        end
    end

    // Packet header fields
    reg [7:0] data_id;
    reg [15:0] word_count;
    reg [7:0] ecc;

    // Virtual channel and data type
    wire [1:0] virtual_channel = data_id[7:6];
    wire [5:0] data_type = data_id[5:0];

    // Frame and line tracking
    reg frame_active;
    reg line_active;
    reg [15:0] current_frame;
    reg [15:0] current_line;

    // Signal hold counters to maintain valid signals for a few cycles
    reg [3:0] frame_valid_counter;
    reg [3:0] line_valid_counter;
    reg [3:0] pixel_valid_counter;
    reg frame_valid_pulse;
    reg frame_valid_int;
    reg short_packet_pending;
    reg [5:0] pending_data_type;
    reg [15:0] pending_word_count;
    reg [1:0] pending_virtual_channel;
    reg [1:0] short_packet_counter;
    assign frame_valid = frame_valid_int;

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= IDLE;
            frame_valid_int <= 1'b0;
            line_valid <= 1'b0;
            pixel_valid <= 1'b0;
            pixel_data <= 8'h00;
            byte_buffer <= 8'h00;
            bit_count <= 3'h0;
            packet_length <= 16'h0;
            byte_count <= 16'h0;
            frame_active <= 1'b0;
            line_active <= 1'b0;
            current_frame <= 16'h0;
            current_line <= 16'h0;
            timeout_counter <= 16'h0;
            frame_valid_counter <= 4'h0;
            line_valid_counter <= 4'h0;
            pixel_valid_counter <= 4'h0;
            frame_valid_pulse <= 1'b0;
            short_packet_pending <= 1'b0;
            pending_data_type <= 6'h00;
            pending_word_count <= 16'h0000;
            pending_virtual_channel <= 2'b00;
            short_packet_counter <= 0;
        end else begin
            case (state)
                IDLE: begin
                    // Wait for HS sync pattern - look for start of packet
                    if (hs_clock_active && hs_data_active) begin
                        state <= HS_SYNC;
                        bit_count <= 3'h0;
                        byte_count <= 0;
                        timeout_counter <= 16'h0;
                        $display("DUT: HS sync detected, entering HS_SYNC state");
                    end
                end

                HS_SYNC: begin
                    // Sample data on HS clock edge and accumulate bits into bytes (MSB first)
                    if (hs_clock_active && hs_data_active) begin  // Only sample when both HS clock and data are active
                        byte_buffer <= {byte_buffer[6:0], data0_lane};
                        bit_count <= bit_count + 1;
                        timeout_counter <= 16'h0; // Reset timeout on valid data
                        //$display("DUT: [HS_SYNC] bit_count=%0d, byte_buffer=0x%02x, data0_lane=%b", bit_count, byte_buffer, data0_lane);
                        if (bit_count == 7) begin
                            // Complete byte received
                            if (byte_count < 4) begin
                                // Header bytes
                                case (byte_count)
                                    0: data_id <= {byte_buffer[6:0], data0_lane};
                                    1: word_count[7:0] <= {byte_buffer[6:0], data0_lane};
                                    2: word_count[15:8] <= {byte_buffer[6:0], data0_lane};
                                    3: ecc <= {byte_buffer[6:0], data0_lane};
                                endcase
                                byte_count <= byte_count + 1;
                                if (byte_count == 3) begin
                                    // Header complete
                                    packet_length <= word_count;
                                    $display("DUT: Header complete - data_id=0x%02x, word_count=%d, data_type=0x%02x", data_id, word_count, data_type);
                                    if (word_count == 0) begin
                                        // Short packet - set pending flag and data
                                        short_packet_pending <= 1'b1;
                                        pending_data_type <= data_type;
                                        pending_word_count <= word_count;
                                        pending_virtual_channel <= virtual_channel;
                                        state <= SHORT_PACKET_PROCESS;
                                    end else begin
                                        // Long packet
                                        state <= PAYLOAD;
                                    end
                                    byte_count <= 0;
                                end
                            end
                            bit_count <= 0;
                            byte_buffer <= 0;
                        end
                    end else begin
                        // Increment timeout counter when no valid HS data
                        timeout_counter <= timeout_counter + 1;
                        if (timeout_counter > 16'h1000) begin // Timeout after ~16us
                            state <= IDLE;
                            $display("DUT: Timeout in HS_SYNC, returning to IDLE");
                        end
                    end
                end

                PAYLOAD: begin
                    // Receive payload data
                    if (hs_clock_active && hs_data_active) begin  // Only sample when both HS clock and data are active
                        byte_buffer <= {byte_buffer[6:0], data0_lane};
                        bit_count <= bit_count + 1;
                        //$display("DUT: [PAYLOAD] bit_count=%0d, byte_buffer=0x%02x, data0_lane=%b", bit_count, byte_buffer, data0_lane);
                        if (bit_count == 7) begin
                            // Complete payload byte
                            pixel_data <= {byte_buffer[6:0], data0_lane};
                            pixel_valid_counter <= 4'h4; // Hold for 4 cycles
                            byte_count <= byte_count + 1;
                            if (byte_count >= packet_length - 1) begin
                                state <= CHECKSUM;
                                byte_count <= 0;
                            end
                            bit_count <= 0;
                            byte_buffer <= 0;
                        end
                    end else begin
                        // Increment timeout counter when no valid HS data
                        timeout_counter <= timeout_counter + 1;
                        if (timeout_counter > 16'h1000) begin // Timeout after ~16us
                            state <= IDLE;
                            $display("DUT: Timeout in PAYLOAD, returning to IDLE");
                        end
                    end
                end

                CHECKSUM: begin
                    // Receive checksum (2 bytes)
                    if (hs_clock_active && hs_data_active) begin  // Only sample when both HS clock and data are active
                        byte_buffer <= {byte_buffer[6:0], data0_lane};
                        bit_count <= bit_count + 1;
                        //$display("DUT: [CHECKSUM] bit_count=%0d, byte_buffer=0x%02x, data0_lane=%b", bit_count, byte_buffer, data0_lane);
                        if (bit_count == 7) begin
                            byte_count <= byte_count + 1;
                            if (byte_count >= 1) begin
                                // Packet complete
                                state <= IDLE;
                                byte_count <= 0;
                            end
                            bit_count <= 0;
                            byte_buffer <= 0;
                        end
                    end else begin
                        // Increment timeout counter when no valid HS data
                        timeout_counter <= timeout_counter + 1;
                        if (timeout_counter > 16'h1000) begin // Timeout after ~16us
                            state <= IDLE;
                            $display("DUT: Timeout in CHECKSUM, returning to IDLE");
                        end
                    end
                end

                SHORT_PACKET_PROCESS: begin
                    // Process pending short packet and assert frame_valid for two full clocks
                    case (pending_data_type)
                        6'h00: begin // Frame Start
                            frame_active <= 1'b1;
                            frame_valid_counter <= 4'hF; // Hold for 16 cycles
                            frame_valid_int <= 1'b1; // Assert immediately
                            current_frame <= pending_word_count;
                            $display("DUT: Frame Start processed - VC=%d, Frame=%d, frame_valid_counter=%d", pending_virtual_channel, pending_word_count, 4'hF);
                        end
                        6'h01: begin // Frame End
                            frame_active <= 1'b0;
                            line_active <= 1'b0;
                            $display("Frame End: VC=%d, Frame=%d", pending_virtual_channel, pending_word_count);
                        end
                        6'h02: begin // Line Start
                            line_active <= 1'b1;
                            line_valid_counter <= 4'h8; // Hold for 8 cycles
                            current_line <= pending_word_count;
                            $display("Line Start: VC=%d, Line=%d", pending_virtual_channel, pending_word_count);
                        end
                        6'h03: begin // Line End
                            line_active <= 1'b0;
                            $display("Line End: VC=%d, Line=%d", pending_virtual_channel, pending_word_count);
                        end
                        6'h08, 6'h09, 6'h0A, 6'h0B, 6'h0C, 6'h0D, 6'h0E, 6'h0F: begin // Generic Short Packets
                            $display("Generic Short Packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                        6'h18, 6'h19, 6'h1A, 6'h1C, 6'h1D, 6'h1E, 6'h1F: begin // YUV Data Types
                            $display("YUV Data Packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                        6'h20, 6'h21, 6'h22, 6'h23, 6'h24: begin // RGB Data Types
                            $display("RGB Data Packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                        6'h28, 6'h29, 6'h2A, 6'h2B, 6'h2C, 6'h2D, 6'h2E, 6'h2F: begin // RAW Data Types
                            $display("RAW Data Packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                        6'h30, 6'h31, 6'h32, 6'h33, 6'h34, 6'h35, 6'h36, 6'h37: begin // User Defined Types
                            $display("User Defined Packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                        default: begin
                            $display("Unknown short packet: DT=0x%02x, VC=%d, Data=0x%04x",
                                    pending_data_type, pending_virtual_channel, pending_word_count);
                        end
                    endcase
                    short_packet_pending <= 1'b0;
                    state <= IDLE;
                end
            endcase

            // Maintain valid signals for a few cycles
            if (frame_valid_counter > 0 || frame_active || frame_valid_pulse) begin
                if (frame_valid_counter > 0)
                    frame_valid_counter <= frame_valid_counter - 1;
                frame_valid_int <= 1'b1;
            end else begin
                frame_valid_int <= 1'b0;
            end
            frame_valid_pulse <= 1'b0;

            if (line_valid_counter > 0) begin
                line_valid_counter <= line_valid_counter - 1;
                line_valid <= 1'b1;
            end else begin
                line_valid <= line_active;
            end

            if (pixel_valid_counter > 0) begin
                pixel_valid_counter <= pixel_valid_counter - 1;
                pixel_valid <= 1'b1;
            end else begin
                pixel_valid <= 1'b0;
            end
        end
    end

    // Clock and data signals are driven by the TX model
    // This DUT only receives and processes the signals

endmodule
