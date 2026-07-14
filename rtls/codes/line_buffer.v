// =============================================================================
// line_buffer.v
// Parametric 1D shift-register line buffer for convolution tap extraction
//
// For a 1D CNN with kernel size K and stride S:
//   - Stores the last K samples of the input activation stream
//   - Each clock (when shift_en=1): shifts in new sample, drops oldest
//   - Output: K parallel taps = act[t], act[t-1], ... act[t-(K-1)]
//
// Parameters:
//   K      : maximum kernel size (must be set to max across all layers = 7)
//   DATA_W : data width in bits (8 for INT8)
//
// The kernel size actually used per layer is controlled by k_size input:
//   k_size = 3 -> only taps[2:0] are valid
//   k_size = 5 -> only taps[4:0] are valid
//   k_size = 7 -> all taps[6:0] valid
//
// Synthesis note:
//   - Infers SRL32 or distributed RAM shift registers on Artix-7
//   - K=7 requires 7 FF stages per channel — maps cleanly to SRL16
//   - No BRAM needed; very small LUT footprint
// =============================================================================

module line_buffer #(
    parameter K      = 7,   // max kernel depth (number of taps)
    parameter DATA_W = 8    // bits per sample (INT8)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    shift_en,   // 1 = accept new sample this cycle
    input  wire                    flush,      // 1 = clear all taps (new row/feature)

    // New input sample
    input  wire signed [DATA_W-1:0] data_in,

    // K parallel output taps
    // tap_out[0] = newest sample (data_in from last shift_en)
    // tap_out[K-1] = oldest sample retained
    output wire signed [DATA_W*K-1:0] tap_out
);

    // Shift register: sr[0] is newest, sr[K-1] is oldest
    reg signed [DATA_W-1:0] sr [0:K-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < K; i = i + 1)
                sr[i] <= {DATA_W{1'b0}};
        end else if (flush) begin
            for (i = 0; i < K; i = i + 1)
                sr[i] <= {DATA_W{1'b0}};
        end else if (shift_en) begin
            // Shift: sr[K-1] <- sr[K-2] <- ... <- sr[0] <- data_in
            sr[0] <= data_in;
            for (i = 1; i < K; i = i + 1)
                sr[i] <= sr[i-1];
        end
    end

    // Pack taps to output bus
    // tap_out[DATA_W*k +: DATA_W] = sr[k]
    genvar gk;
    generate
        for (gk = 0; gk < K; gk = gk + 1) begin : pack_taps
            assign tap_out[DATA_W*gk +: DATA_W] = sr[gk];
        end
    endgenerate

endmodule
