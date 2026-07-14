// =============================================================================
// mac_unit.v
// Parallel MAC unit: P simultaneous multiply-accumulate operations
//
// For each of the P output channels, one DSP48E1 computes:
//   acc[p] += activation_tap[k] * weight[p][k]
// across all kernel taps k = 0..K-1, then across all input channels.
//
// Parameters:
//   P          : number of parallel output channels (default 8, tunable)
//   ACC_W      : accumulator width in bits (32 safe for INT8 x INT8 x ~2M taps)
//
// Datapath:
//   - act_in   : K input taps from line_buffer (INT8, sign-extended to 18b)
//   - wgt_in   : P*K weights from weight_bram  (INT8, sign-extended to 18b)
//   - acc_in   : P accumulators fed back or zeroed at start of pixel
//   - mac_out  : P updated accumulators (before ReLU)
//   - relu_out : P values after max(0, x) + INT8 saturation
//
// Pipeline latency: 3 cycles (matches DSP48E1 AREG+MREG+PREG chain)
// For a single kernel tap: output valid after 3 clocks.
// For K taps: accumulate over K cycles; result ready at cycle K+2.
//
// ReLU + Saturation:
//   - max(0, x)           : zero negatives
//   - clamp to [0, 127]   : INT8 output range (unsigned after ReLU)
//   - Implemented in pure LUT logic (no DSP needed)
// =============================================================================

module mac_unit #(
    parameter P     = 8,    // parallel output channels
    parameter K     = 7,    // max kernel size (set per-layer via k_size port)
    parameter ACC_W = 32    // accumulator width
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         ce,

    // Control
    input  wire [2:0]                   k_size,   // actual kernel size: 3, 5, or 7
    input  wire                         acc_clr,  // 1 = zero accumulators (new pixel)
    input  wire                         acc_en,   // 1 = accumulate this cycle
    input  wire                         out_valid_in, // drives output valid pipeline

    // Activation taps from line_buffer (K taps, each INT8)
    input  wire signed [8*K-1:0]        act_taps, // packed: tap[0]=bits[7:0], etc.

    // Weights: P output channels x K taps, each INT8, from weight_bram
    // Packed: wgt[p][k] = wgt_packed[p*K*8 + k*8 +: 8]
    input  wire signed [8*P*K-1:0]      wgt_packed,

    // Bias: P values, INT16 (post-BN fold), added after full accumulation
    input  wire signed [16*P-1:0]       bias_packed,

    // Outputs
    output reg  signed [8*P-1:0]        relu_out,  // packed INT8, one per out-ch
    output reg                          out_valid
);

    // -------------------------------------------------------------------------
    // Unpack inputs into arrays for readability
    // -------------------------------------------------------------------------
    wire signed [7:0]  act  [0:K-1];
    wire signed [7:0]  wgt  [0:P-1][0:K-1];
    wire signed [15:0] bias [0:P-1];

    genvar gp, gk;
    generate
        for (gk = 0; gk < K; gk = gk + 1) begin : unpack_act
            assign act[gk] = act_taps[gk*8 +: 8];
        end
        for (gp = 0; gp < P; gp = gp + 1) begin : unpack_wgt_outer
            for (gk = 0; gk < K; gk = gk + 1) begin : unpack_wgt_inner
                assign wgt[gp][gk] = wgt_packed[(gp*K + gk)*8 +: 8];
            end
            assign bias[gp] = bias_packed[gp*16 +: 16];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Accumulators: one per output channel
    // Depth = ACC_W bits to avoid overflow across many input channels
    // -------------------------------------------------------------------------
    reg signed [ACC_W-1:0] acc [0:P-1];

    // -------------------------------------------------------------------------
    // DSP48E1 wrappers: one per output channel
    // Each DSP computes: act[k] * wgt[p][k] + acc[p]
    // We feed one tap per cycle, accumulating across k in software (via acc_en)
    //
    // For simplicity, we instantiate P DSPs and each cycle multiply ONE tap.
    // The tap index is driven by a counter in conv_engine.
    // conv_engine presents one (act_tap, wgt_col) pair per cycle.
    // -------------------------------------------------------------------------

    // Current tap index driven externally via acc_clr/acc_en sequence
    // We use a local tap counter reset by acc_clr
    reg [2:0] tap_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            tap_cnt <= 3'd0;
        else if (acc_clr)
            tap_cnt <= 3'd0;
        else if (acc_en && tap_cnt < k_size - 1)
            tap_cnt <= tap_cnt + 1'd1;
        else if (acc_en)
            tap_cnt <= 3'd0;  // wraps at end of kernel (multi-channel handled by conv_engine)
    end

    // DSP outputs: P channels
    wire signed [47:0] dsp_p [0:P-1];

    // Sign-extend act tap and weight for DSP
    wire signed [17:0] act_dsp;
    wire signed [17:0] wgt_dsp [0:P-1];

    assign act_dsp = {{10{act[tap_cnt][7]}}, act[tap_cnt]};

    generate
        for (gp = 0; gp < P; gp = gp + 1) begin : dsp_chain
            assign wgt_dsp[gp] = {{10{wgt[gp][tap_cnt][7]}}, wgt[gp][tap_cnt]};

            // C port: feed current accumulator (48-bit sign-extended)
            wire signed [47:0] acc_feed;
            assign acc_feed = acc_clr ? 48'sd0 :
                              {{(48-ACC_W){acc[gp][ACC_W-1]}}, acc[gp]};

            dsp48e1_wrap dsp_inst (
                .clk   (clk),
                .rst_n (rst_n),
                .ce    (ce & acc_en),
                .A     (act_dsp),
                .B     (wgt_dsp[gp]),
                .C     (acc_feed),
                .P     (dsp_p[gp])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Accumulator update
    // DSP has 3-cycle latency, so we use a shift register to track when to
    // write DSP output back to accumulator.
    // -------------------------------------------------------------------------
    reg [2:0] acc_en_pipe;   // 3-bit pipeline for acc_en
    reg [2:0] acc_clr_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_en_pipe  <= 3'b0;
            acc_clr_pipe <= 3'b0;
        end else if (ce) begin
            acc_en_pipe  <= {acc_en_pipe[1:0],  acc_en};
            acc_clr_pipe <= {acc_clr_pipe[1:0], acc_clr};
        end
    end

    generate
        for (gp = 0; gp < P; gp = gp + 1) begin : acc_update
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    acc[gp] <= {ACC_W{1'b0}};
                else if (ce) begin
                    if (acc_clr_pipe[2])
                        // Clear with first DSP result (clr happened 3 cycles ago)
                        acc[gp] <= dsp_p[gp][ACC_W-1:0];
                    else if (acc_en_pipe[2])
                        acc[gp] <= dsp_p[gp][ACC_W-1:0];
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Bias addition + ReLU + INT8 saturation
    // Triggered by out_valid_in (pulsed by conv_engine when all k,cin done)
    // -------------------------------------------------------------------------
    reg out_valid_pipe;
reg signed [32:0] biased;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_pipe <= 1'b0;
            out_valid      <= 1'b0;
            relu_out       <= {8*P{1'b0}};
        end else if (ce) begin
            out_valid_pipe <= out_valid_in;
            out_valid      <= out_valid_pipe;

            if (out_valid_pipe) begin : relu_bias
                integer p;
                for (p = 0; p < P; p = p + 1) begin
                    // Add folded BN bias (INT16) to accumulator
                    // Shift right by FRAC_BITS if using fixed-point scale
                    // Here we assume weights already in INT8 scale, acc in INT32
                    // Bias is INT16; sum fits in 33 bits
                    biased = {{(33-ACC_W){acc[p][ACC_W-1]}}, acc[p]}
                             + {{17{bias[p][15]}}, bias[p]};

                    // ReLU: clamp negative to 0
                    // Saturate to [0, 127] for INT8 unsigned output
                    if (biased <= 0)
                        relu_out[p*8 +: 8] <= 8'd0;
                    else if (biased > 32'sd127)
                        relu_out[p*8 +: 8] <= 8'd127;
                    else
                        relu_out[p*8 +: 8] <= biased[7:0];
                end
            end
        end
    end

endmodule
