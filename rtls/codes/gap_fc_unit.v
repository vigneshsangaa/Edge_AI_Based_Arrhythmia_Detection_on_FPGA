// =============================================================================
// gap_fc_unit.v
// Global Average Pool → Fully Connected → Argmax
//
// After Conv4, activations are (64 channels × 24 time steps).
// This unit:
//   1. GAP  : for each of 64 channels, sum 24 values, shift right 5 (≈ ÷24)
//             Using >>5 (÷32) introduces ~25% error vs true ÷24.
//             For better accuracy use a 5-bit right-shift with rounding:
//             gap[c] = (sum[c] + 12) >> 5  (add half-divisor before shift)
//             Or use true division: sum / 24 via small LUT divider (24=8×3).
//             We use the shift approximation here; replace with divider if needed.
//
//   2. FC   : 64-input × 5-output linear layer (weights INT8, bias INT16)
//             5 parallel DSPs, 64 cycles per output → 64 cycles total
//             (all 5 outputs accumulate simultaneously)
//
//   3. Argmax: find index of maximum among 5 INT32 logits
//             Pure combinational comparison tree
//
// Ports:
//   act_bram  : read interface to activation BRAM (post-Conv4 bank)
//   wgt_fc    : FC weights, pre-loaded from weight BRAM (5×64 INT8)
//   bias_fc   : FC bias, 5×INT16
//   pred_class: 3-bit output (0..4) - predicted arrhythmia class
// =============================================================================

// =============================================================================
// gap_fc_unit.v
// Weights and biases read from external BRAMs (Fc_W, Fc_B) via address ports.
// =============================================================================

module gap_fc_unit #(
    parameter CH     = 64,
    parameter T      = 24,
    parameter N_CLS  = 5,
    parameter DATA_W = 8,
    parameter ACC_W  = 32,
    parameter ADDR_W = 16
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [2:0]  pred_class,

    // Activation BRAM read
    output reg  [ADDR_W-1:0] act_rd_addr,
    output reg               act_rd_en,
    input  wire [DATA_W-1:0] act_rd_data,

    // FC Weight BRAM read (Fc_W: 320 entries, INT8)
    output reg  [8:0]               fc_wgt_addr,   // 9-bit: 0..319
    output reg                      fc_wgt_en,
    input  wire signed [DATA_W-1:0] fc_wgt_data,   // one INT8 per cycle

    // FC Bias BRAM read (Fc_B: 5 entries, INT16)
    output reg  [2:0]          fc_bias_addr,  // 3-bit: 0..4
    output reg                 fc_bias_en,
    input  wire signed [15:0]  fc_bias_data   // one INT16 per cycle
);

    // =========================================================================
    // Internal storage
    // =========================================================================
    reg signed [DATA_W-1:0] gap_out  [0:CH-1];    // GAP results
    reg signed [ACC_W-1:0]  gap_acc  [0:CH-1];    // GAP accumulators
    reg signed [ACC_W-1:0]  fc_acc   [0:N_CLS-1]; // FC accumulators
    reg signed [ACC_W-1:0]  logit    [0:N_CLS-1]; // logits after bias

    // FC weight buffer: all 320 weights loaded before MAC loop
    reg signed [DATA_W-1:0] wgt_buf  [0:CH*N_CLS-1];

    // =========================================================================
    // Counters
    // =========================================================================
    reg [6:0]  ch_cnt;    // 0..CH-1  (GAP channel counter)
    reg [4:0]  t_cnt;     // 0..T-1   (GAP time counter)
    reg [6:0]  fc_ch;     // 0..CH-1  (FC input channel counter)
    reg [2:0]  fc_cls;    // 0..N_CLS-1 (FC class counter)
    reg [8:0]  wld_cnt;   // 0..319   (weight load counter)
    reg [2:0]  bias_cnt;  // 0..4     (bias load counter)

    // =========================================================================
    // Argmax (combinational)
    // =========================================================================
    reg [2:0]              argmax_result;
    reg signed [ACC_W-1:0] argmax_val;
    integer j;

    always @(*) begin
        argmax_result = 3'd0;
        argmax_val    = logit[0];
        for (j = 1; j < N_CLS; j = j + 1) begin
            if (logit[j] > argmax_val) begin
                argmax_val    = logit[j];
                argmax_result = j[2:0];
            end
        end
    end

    // =========================================================================
    // FSM states
    // =========================================================================
    localparam S_IDLE       = 4'd0;
    localparam S_GAP_READ   = 4'd1;
    localparam S_GAP_WAIT   = 4'd2;
    localparam S_GAP_SHIFT  = 4'd3;
    localparam S_WGT_LOAD   = 4'd4;
    localparam S_WGT_WAIT   = 4'd5;
    localparam S_FC_RUN     = 4'd6;
    localparam S_BIAS_LOAD  = 4'd7;
    localparam S_BIAS_WAIT  = 4'd8;
    localparam S_BIAS_ACC   = 4'd9;
    localparam S_ARGMAX     = 4'd10;
    localparam S_DONE       = 4'd11;

    reg [3:0] state;
    integer   i;
    reg signed [ACC_W-1:0] rounded;

    // REQP-1839/1840: this block drives BRAM control pins (act_rd_en,
    // fc_wgt_en, fc_bias_en) so it must use a synchronous reset (FDRE),
    // not an asynchronous one.
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            done        <= 1'b0;
            ch_cnt      <= 7'd0;
            t_cnt       <= 5'd0;
            fc_ch       <= 7'd0;
            fc_cls      <= 3'd0;
            wld_cnt     <= 9'd0;
            bias_cnt    <= 3'd0;
            act_rd_en   <= 1'b0;
            fc_wgt_en   <= 1'b0;
            fc_bias_en  <= 1'b0;
            for (i = 0; i < CH;    i = i + 1) gap_acc[i] <= 0;
            for (i = 0; i < CH;    i = i + 1) gap_out[i] <= 0;
            for (i = 0; i < N_CLS; i = i + 1) fc_acc[i]  <= 0;
            for (i = 0; i < N_CLS; i = i + 1) logit[i]   <= 0;
        end else begin
            done       <= 1'b0;
            act_rd_en  <= 1'b0;
            fc_wgt_en  <= 1'b0;
            fc_bias_en <= 1'b0;

            case (state)

            S_IDLE: begin
                if (start) begin
                    ch_cnt <= 7'd0;
                    t_cnt  <= 5'd0;
                    for (i = 0; i < CH; i = i + 1) gap_acc[i] <= 0;
                    state  <= S_GAP_READ;
                end
            end

            // Issue read address; absorb BRAM latency in S_GAP_WAIT
            S_GAP_READ: begin
                act_rd_addr <= ch_cnt * T[ADDR_W-1:0] + {{(ADDR_W-5){1'b0}}, t_cnt};
                act_rd_en   <= 1'b1;
                state       <= S_GAP_WAIT;
            end

            // Data valid this cycle (1-cycle BRAM latency)
            S_GAP_WAIT: begin
                gap_acc[ch_cnt] <= gap_acc[ch_cnt]
                                   + {{(ACC_W-DATA_W){act_rd_data[DATA_W-1]}},
                                      act_rd_data};
                if (t_cnt == T[4:0] - 1) begin
                    t_cnt <= 5'd0;
                    if (ch_cnt == CH[6:0] - 1) begin
                        ch_cnt <= 7'd0;
                        state  <= S_GAP_SHIFT;
                    end else begin
                        ch_cnt <= ch_cnt + 1'd1;
                        state  <= S_GAP_READ;
                    end
                end else begin
                    t_cnt <= t_cnt + 1'd1;
                    state <= S_GAP_READ;
                end
            end

            // Divide-by-24 via >>5 with rounding, clamp to [0,127]
            S_GAP_SHIFT: begin
                for (i = 0; i < CH; i = i + 1) begin
                    rounded = (gap_acc[i] + 12) >>> 5;
                    if      (rounded < 0)   gap_out[i] <= 8'sd0;
                    else if (rounded > 127) gap_out[i] <= 8'sd127;
                    else                    gap_out[i] <= rounded[DATA_W-1:0];
                end
                wld_cnt <= 9'd0;
                state   <= S_WGT_LOAD;
            end

            // Load all 320 FC weights from Fc_W BRAM sequentially
            S_WGT_LOAD: begin
                fc_wgt_addr <= wld_cnt;
                fc_wgt_en   <= 1'b1;
                if (wld_cnt == 9'd0) begin
                    wld_cnt <= wld_cnt + 1'd1;
                    state   <= S_WGT_LOAD;
                end else begin
                    wgt_buf[wld_cnt - 1] <= fc_wgt_data;
                    if (wld_cnt == CH * N_CLS) begin
                        state <= S_WGT_WAIT;
                    end else begin
                        wld_cnt <= wld_cnt + 1'd1;
                    end
                end
            end

            // Capture final weight word, reset FC accumulators
            S_WGT_WAIT: begin
                wgt_buf[CH * N_CLS - 1] <= fc_wgt_data;
                for (i = 0; i < N_CLS; i = i + 1) fc_acc[i] <= 0;
                fc_ch  <= 7'd0;
                fc_cls <= 3'd0;
                state  <= S_FC_RUN;
            end

            // FC MAC: all CH channels for current class
            // wgt_buf layout: wgt_buf[cls*CH + ch]
            S_FC_RUN: begin
                fc_acc[fc_cls] <= fc_acc[fc_cls]
                    + gap_out[fc_ch] * wgt_buf[fc_cls * CH[6:0] + fc_ch];

                if (fc_ch == CH[6:0] - 1) begin
                    fc_ch <= 7'd0;
                    if (fc_cls == N_CLS[2:0] - 1) begin
                        bias_cnt     <= 3'd0;
                        fc_bias_addr <= 3'd0;
                        fc_bias_en   <= 1'b1;
                        state        <= S_BIAS_WAIT;
                    end else begin
                        fc_cls <= fc_cls + 1'd1;
                    end
                end else begin
                    fc_ch <= fc_ch + 1'd1;
                end
            end

            // Absorb BRAM read latency for bias
            S_BIAS_WAIT: begin
                state <= S_BIAS_ACC;
            end

            // Add bias to fc_acc, request next bias
            S_BIAS_ACC: begin
                logit[bias_cnt] <= fc_acc[bias_cnt]
                                   + {{(ACC_W-16){fc_bias_data[15]}}, fc_bias_data};
                if (bias_cnt == N_CLS[2:0] - 1) begin
                    state <= S_ARGMAX;
                end else begin
                    bias_cnt     <= bias_cnt + 1'd1;
                    fc_bias_addr <= bias_cnt + 1'd1;
                    fc_bias_en   <= 1'b1;
                    state        <= S_BIAS_WAIT;
                end
            end

            S_ARGMAX: begin
                pred_class <= argmax_result;
                state      <= S_DONE;
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule