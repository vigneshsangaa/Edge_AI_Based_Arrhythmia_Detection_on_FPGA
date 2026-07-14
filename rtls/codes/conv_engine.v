// =============================================================================
// conv_engine.v  (FIXED)
// Reusable 1D convolution engine - shared across Conv1..Conv4
//
// CHANGES vs original:
//   1. BUG FIX: mac_bias was declared but never driven, and bias_rd_data was
//      never consumed -> every layer ran with bias = 0. Added a S_BIAS_LOAD
//      state that sequentially reads P biases from the bias BRAM (one INT16
//      per cycle, 1-cycle read latency) and packs them into mac_bias before
//      triggering the MAC unit's bias-add/ReLU pipeline.
//   2. Removed dead code: wr_base register was set but never read anywhere
//      (act_wr_addr is computed directly from cg_cnt/t_pos), so it was
//      being optimized away by synthesis. Removed for a clean report.
//
// Architecture:
//   - ONE instance of this module is reused for all 4 layers sequentially
//   - Configured per-layer via parameter ports (Cin, Cout, kernel, stride, length)
//   - Reads activations from act_bram (ping-pong, current read bank)
//   - Reads weights from weight_bram (pre-loaded for current layer)
//   - Writes results to act_bram (current write bank)
//
// Computation order (innermost-first):
//   for each output position t (0..L_out-1):
//     for each output channel group g (0..Cout/P-1):
//       for each input channel cin (0..Cin-1):
//         load K taps from line_buffer
//         mac_unit computes P channels simultaneously
//       after all cin: load P biases, then bias + relu -> write P values
//
// DSP utilisation:
//   P=8 DSPs active per cycle.
//   Layers with Cout not divisible by P: pad Cout to next multiple of P.
//   Extra outputs are simply discarded (zero-bias keeps them 0).
//
// Ports:
//   All data busses use packed INT8 (8 bits per value).
//   Addresses are word-addressed (one word = one INT8 activation).
//
// NOTE: bias BRAM must store biases contiguously as [cg*P + p] (P INT16
//   values per output-channel group), matching bias_rd_addr generation below.
// =============================================================================

module conv_engine #(
    parameter P      = 8,    // parallel output channels per cycle
    parameter K_MAX  = 7,    // max kernel size
    parameter DATA_W = 8,    // INT8
    parameter ACC_W  = 32,
    parameter ADDR_W = 16    // enough for max activation size
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,      // pulse to begin processing
    output reg         done,       // pulses high for 1 cycle when complete

    // Layer configuration (set by fsm_controller before start)
    input  wire [7:0]  cfg_cin,    // input channels
    input  wire [7:0]  cfg_cout,   // output channels (must be multiple of P)
    input  wire [2:0]  cfg_ksize,  // kernel size: 3, 5, or 7
    input  wire [1:0]  cfg_stride, // stride: 1 or 2
    input  wire [8:0]  cfg_lin,    // input length (max 256 for this design)
    input  wire [8:0]  cfg_lout,   // output length
    input  wire [1:0]  cfg_pad,    // same-padding amount each side

    // Activation BRAM interface (read side - current input bank)
    output reg  [ADDR_W-1:0] act_rd_addr,
    output reg               act_rd_en,
    input  wire [DATA_W-1:0] act_rd_data,   // 1 byte per cycle

    // Activation BRAM interface (write side - current output bank)
    output reg  [ADDR_W-1:0] act_wr_addr,
    output reg               act_wr_en,
    output reg  [DATA_W-1:0] act_wr_data,

    // Weight BRAM interface (read side)
    // Address scheme: [cout_group][cin][tap]
    output reg  [ADDR_W-1:0] wgt_rd_addr,
    output reg               wgt_rd_en,
    input  wire [DATA_W*P*K_MAX-1:0] wgt_rd_data,  // P*K weights in one read

    // Bias BRAM interface (one bias per output channel, INT16)
    output reg  [ADDR_W-1:0] bias_rd_addr,
    output reg               bias_rd_en,
    input  wire [15:0]       bias_rd_data   // one INT16 bias per read
);

    // =========================================================================
    // Internal state
    // =========================================================================

    // Loop counters
    reg [8:0]  t_pos;     // output position counter  0..cfg_lout-1
    reg [7:0]  cin_cnt;   // input channel counter     0..cfg_cin-1
    reg [2:0]  tap_cnt;   // kernel tap counter        0..cfg_ksize-1
    reg [7:0]  cg_cnt;    // output channel group      0..Cout/P-1
    reg [3:0]  bias_ld_cnt; // bias load counter        0..P

    // One line_buffer per input channel would be ideal but too many FFs.
    // Instead: we process ONE input channel at a time, loading its segment
    // from BRAM into a single line_buffer and streaming through.
    // This trades throughput for area (suitable for Artix-7 budget).

    wire signed [DATA_W*K_MAX-1:0] lb_taps;
    reg  lb_shift_en, lb_flush;
    reg signed [DATA_W-1:0] lb_data_in;

    line_buffer #(.K(K_MAX), .DATA_W(DATA_W)) lb_inst (
        .clk       (clk),
        .rst_n     (rst_n),
        .shift_en  (lb_shift_en),
        .flush     (lb_flush),
        .data_in   (lb_data_in),
        .tap_out   (lb_taps)
    );

    // =========================================================================
    // MAC unit
    // =========================================================================

    wire signed [DATA_W*P-1:0] mac_relu_out;
    wire                       mac_out_valid;

    reg                        mac_acc_clr;
    reg                        mac_acc_en;
    reg                        mac_out_valid_in;
    reg signed [DATA_W*P*K_MAX-1:0] mac_wgt;
    reg signed [16*P-1:0]           mac_bias;

    mac_unit #(.P(P), .K(K_MAX), .ACC_W(ACC_W)) mac_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .ce           (1'b1),
        .k_size       (cfg_ksize),
        .acc_clr      (mac_acc_clr),
        .acc_en       (mac_acc_en),
        .out_valid_in (mac_out_valid_in),
        .act_taps     (lb_taps),
        .wgt_packed   (mac_wgt),
        .bias_packed  (mac_bias),
        .relu_out     (mac_relu_out),
        .out_valid    (mac_out_valid)
    );

    // =========================================================================
    // Output write tracking
    // =========================================================================
    reg [2:0]        wr_sub;     // sub-index within P outputs (0..P-1)

    // =========================================================================
    // Main FSM
    // =========================================================================
    localparam S_IDLE        = 4'd0;
    localparam S_SETUP       = 4'd1;  // latch config, reset counters
    localparam S_LOAD_ACT    = 4'd2;  // stream activation into line_buffer
    localparam S_FILL_LB     = 4'd3;  // wait for line_buffer to fill K taps
    localparam S_MAC_RUN     = 4'd4;  // accumulate K taps for one cin, one cg
    localparam S_NEXT_CIN    = 4'd5;  // advance input channel
    localparam S_NEXT_CG     = 4'd6;  // advance output channel group
    localparam S_WRITE_OUT   = 4'd7;  // write P outputs to BRAM
    localparam S_NEXT_T      = 4'd8;  // advance output position
    localparam S_DONE        = 4'd9;
    localparam S_BIAS_LOAD   = 4'd10; // load P biases before bias-add/ReLU

    reg [3:0] state;
    reg [3:0] fill_cnt;   // counts K cycles during line_buffer fill

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            t_pos           <= 9'd0;
            cin_cnt         <= 8'd0;
            cg_cnt          <= 8'd0;
            tap_cnt         <= 3'd0;
            fill_cnt        <= 4'd0;
            bias_ld_cnt     <= 4'd0;
            mac_bias        <= {(16*P){1'b0}};
            mac_acc_clr     <= 1'b0;
            mac_acc_en      <= 1'b0;
            mac_out_valid_in<= 1'b0;
            lb_flush        <= 1'b0;
            lb_shift_en     <= 1'b0;
            act_rd_en       <= 1'b0;
            act_wr_en       <= 1'b0;
            wgt_rd_en       <= 1'b0;
            bias_rd_en      <= 1'b0;
        end else begin
            done            <= 1'b0;
            mac_acc_clr     <= 1'b0;
            mac_acc_en      <= 1'b0;
            mac_out_valid_in<= 1'b0;
            lb_flush        <= 1'b0;
            lb_shift_en     <= 1'b0;
            act_rd_en       <= 1'b0;
            act_wr_en       <= 1'b0;
            wgt_rd_en       <= 1'b0;
            bias_rd_en      <= 1'b0;

            case (state)

            // -----------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    t_pos   <= 9'd0;
                    cin_cnt <= 8'd0;
                    cg_cnt  <= 8'd0;
                    state   <= S_SETUP;
                end
            end

            // -----------------------------------------------------------------
            // S_SETUP: flush line buffer
            S_SETUP: begin
                lb_flush    <= 1'b1;
                mac_acc_clr <= 1'b1;   // zero accumulators
                state       <= S_LOAD_ACT;
            end

            // -----------------------------------------------------------------
            // S_LOAD_ACT: stream cfg_lin activations for current cin into lb
            // Address = cin_cnt * cfg_lin + (t_pos * stride) - pad + tap offset
            // Simplified: load K consecutive samples starting at tap window
            // For padding: if address < 0 or >= cfg_lin, feed 0 instead
            S_LOAD_ACT: begin
                // Stream K samples into line buffer for current output position
                // t_pos output corresponds to input range [t_pos*S - pad, +K)
                act_rd_addr <= cin_cnt * cfg_lin
                               + (t_pos * cfg_stride)
                               - cfg_pad
                               + fill_cnt;
                act_rd_en   <= 1'b1;
                fill_cnt    <= fill_cnt + 1'd1;

                if (fill_cnt == cfg_ksize - 1) begin
                    fill_cnt <= 4'd0;
                    state    <= S_FILL_LB;
                end
            end

            // -----------------------------------------------------------------
            // S_FILL_LB: BRAM has 1-cycle read latency; wait then load lb
            S_FILL_LB: begin
                lb_data_in  <= act_rd_data;
                lb_shift_en <= 1'b1;
                // Load weight for current cg and cin
                wgt_rd_addr <= cg_cnt * cfg_cin * K_MAX
                               + cin_cnt * K_MAX;
                wgt_rd_en   <= 1'b1;
                state       <= S_MAC_RUN;
            end

            // -----------------------------------------------------------------
            // S_MAC_RUN: run MAC for cfg_ksize taps
            S_MAC_RUN: begin
                mac_wgt    <= wgt_rd_data;
                mac_acc_en <= 1'b1;
                tap_cnt    <= tap_cnt + 1'd1;

                if (tap_cnt == cfg_ksize - 1) begin
                    tap_cnt <= 3'd0;
                    state   <= S_NEXT_CIN;
                end
            end

            // -----------------------------------------------------------------
            // S_NEXT_CIN: advance input channel, or start bias load if all
            // cin done for this output-channel group
            S_NEXT_CIN: begin
                if (cin_cnt < cfg_cin - 1) begin
                    cin_cnt <= cin_cnt + 1'd1;
                    state   <= S_LOAD_ACT;
                end else begin
                    cin_cnt      <= 8'd0;
                    bias_ld_cnt  <= 4'd0;
                    // Base address of this output group's P biases:
                    // bias BRAM layout is [cg*P + p], p = 0..P-1
                    bias_rd_addr <= cg_cnt * P[ADDR_W-1:0];
                    bias_rd_en   <= 1'b1;
                    state        <= S_BIAS_LOAD;
                end
            end

            // -----------------------------------------------------------------
            // S_BIAS_LOAD: sequentially read P biases (1-cycle BRAM latency)
            // and pack them into mac_bias. Data for address issued on cycle N
            // is valid on cycle N+1, so we capture bias_rd_data one cycle
            // after each address is issued (bias_ld_cnt counts addresses
            // issued; captured data lags by one).
            S_BIAS_LOAD: begin
                if (bias_ld_cnt > 0)
                    mac_bias[(bias_ld_cnt-1)*16 +: 16] <= bias_rd_data;

                if (bias_ld_cnt == P[3:0]) begin
                    mac_out_valid_in <= 1'b1;
                    state            <= S_WRITE_OUT;
                end else begin
                    bias_rd_addr <= cg_cnt * P[ADDR_W-1:0] + bias_ld_cnt;
                    bias_rd_en   <= 1'b1;
                    bias_ld_cnt  <= bias_ld_cnt + 1'd1;
                end
            end

            // -----------------------------------------------------------------
            // S_WRITE_OUT: wait for MAC pipeline drain then write P results
            S_WRITE_OUT: begin
                // mac_out_valid arrives 3 cycles after mac_out_valid_in
                // (handled by mac_unit internal pipeline)
                if (mac_out_valid) begin
                    // Write P outputs sequentially to BRAM
                    // Output address layout: [cout_ch][t_pos]
                    // = (cg_cnt*P + wr_sub) * cfg_lout + t_pos
                    act_wr_addr <= (cg_cnt * P[ADDR_W-1:0] + wr_sub)
                                   * cfg_lout + t_pos;
                    act_wr_data <= mac_relu_out[wr_sub*8 +: 8];
                    act_wr_en   <= 1'b1;
                    wr_sub      <= wr_sub + 1'd1;

                    if (wr_sub == P[2:0] - 1) begin
                        wr_sub  <= 3'd0;
                        state   <= S_NEXT_CG;
                    end
                end
            end

            // -----------------------------------------------------------------
            // S_NEXT_CG: advance output channel group
            S_NEXT_CG: begin
                if (cg_cnt < (cfg_cout >> $clog2(P)) - 1) begin
                    cg_cnt      <= cg_cnt + 1'd1;
                    cin_cnt     <= 8'd0;
                    mac_acc_clr <= 1'b1;
                    state       <= S_LOAD_ACT;
                end else begin
                    cg_cnt  <= 8'd0;
                    state   <= S_NEXT_T;
                end
            end

            // -----------------------------------------------------------------
            // S_NEXT_T: advance output position
            S_NEXT_T: begin
                if (t_pos < cfg_lout - 1) begin
                    t_pos       <= t_pos + 1'd1;
                    cin_cnt     <= 8'd0;
                    cg_cnt      <= 8'd0;
                    lb_flush    <= 1'b1;
                    mac_acc_clr <= 1'b1;
                    state       <= S_LOAD_ACT;
                end else begin
                    state <= S_DONE;
                end
            end

            // -----------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule