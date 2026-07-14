module fsm_controller (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        trigger,    // pulse to start inference on one ECG beat
    output reg         busy,       // high during inference
    output reg         valid_out,  // pulses when pred_class is stable
    output reg  [2:0]  pred_class, // 0=N, 1=S, 2=V, 3=F, 4=Q

    // conv_engine control
    output reg         conv_start,
    input  wire        conv_done,
    output reg  [7:0]  conv_cfg_cin,
    output reg  [7:0]  conv_cfg_cout,
    output reg  [2:0]  conv_cfg_ksize,
    output reg  [1:0]  conv_cfg_stride,
    output reg  [8:0]  conv_cfg_lin,
    output reg  [8:0]  conv_cfg_lout,
    output reg  [1:0]  conv_cfg_pad,

    // gap_fc_unit control
    output reg         gap_start,
    input  wire        gap_done,
    input  wire [2:0]  gap_pred,

    // BRAM bank select (ping-pong)
    output reg         act_bank_sel,   // 0=bank_A is input, 1=bank_B is input
    output reg  [1:0]  wgt_layer_sel   // selects which weight bank: 0..3
);

    localparam S_IDLE   = 4'd0;
    localparam S_CONV1  = 4'd1;
    localparam S_SWAP1  = 4'd2;
    localparam S_CONV2  = 4'd3;
    localparam S_SWAP2  = 4'd4;
    localparam S_CONV3  = 4'd5;
    localparam S_SWAP3  = 4'd6;
    localparam S_CONV4  = 4'd7;
    localparam S_SWAP4  = 4'd8;
    localparam S_GAPFC  = 4'd9;
    localparam S_OUTPUT = 4'd10;

    reg [3:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            valid_out      <= 1'b0;
            conv_start     <= 1'b0;
            gap_start      <= 1'b0;
            act_bank_sel   <= 1'b0;
            wgt_layer_sel  <= 2'd0;
            pred_class     <= 3'd0;
        end else begin
            conv_start <= 1'b0;
            gap_start  <= 1'b0;
            valid_out  <= 1'b0;

            case (state)

            S_IDLE: begin
                busy <= 1'b0;
                if (trigger) begin
                    busy         <= 1'b1;
                    act_bank_sel <= 1'b0;   // input ECG in bank A
                    wgt_layer_sel<= 2'd0;   // Conv1 weights
                    // Configure Conv1
                    conv_cfg_cin    <= 8'd1;
                    conv_cfg_cout   <= 8'd32;
                    conv_cfg_ksize  <= 3'd7;
                    conv_cfg_stride <= 2'd2;
                    conv_cfg_lin    <= 9'd187;
                    conv_cfg_lout   <= 9'd94;
                    conv_cfg_pad    <= 2'd3;
                    conv_start      <= 1'b1;
                    state           <= S_CONV1;
                end
            end

            S_CONV1: begin
                if (conv_done) state <= S_SWAP1;
            end

            S_SWAP1: begin
                // Output of Conv1 in bank B (bank_sel was 0 -> write to B)
                act_bank_sel  <= 1'b1;   // now bank B is input
                wgt_layer_sel <= 2'd1;   // Conv2 weights
                conv_cfg_cin    <= 8'd32;
                conv_cfg_cout   <= 8'd64;
                conv_cfg_ksize  <= 3'd5;
                conv_cfg_stride <= 2'd2;
                conv_cfg_lin    <= 9'd94;
                conv_cfg_lout   <= 9'd47;
                conv_cfg_pad    <= 2'd2;
                conv_start      <= 1'b1;
                state           <= S_CONV2;
            end

            S_CONV2: begin
                if (conv_done) state <= S_SWAP2;
            end

            S_SWAP2: begin
                act_bank_sel  <= 1'b0;   // back to bank A for output
                wgt_layer_sel <= 2'd2;
                conv_cfg_cin    <= 8'd64;
                conv_cfg_cout   <= 8'd128;
                conv_cfg_ksize  <= 3'd3;
                conv_cfg_stride <= 2'd2;
                conv_cfg_lin    <= 9'd47;
                conv_cfg_lout   <= 9'd24;
                conv_cfg_pad    <= 2'd1;
                conv_start      <= 1'b1;
                state           <= S_CONV3;
            end

            S_CONV3: begin
                if (conv_done) state <= S_SWAP3;
            end

            S_SWAP3: begin
                act_bank_sel  <= 1'b1;
                wgt_layer_sel <= 2'd3;
                conv_cfg_cin    <= 8'd128;
                conv_cfg_cout   <= 8'd64;
                conv_cfg_ksize  <= 3'd3;
                conv_cfg_stride <= 2'd1;
                conv_cfg_lin    <= 9'd24;
                conv_cfg_lout   <= 9'd24;
                conv_cfg_pad    <= 2'd1;
                conv_start      <= 1'b1;
                state           <= S_CONV4;
            end

            S_CONV4: begin
                if (conv_done) state <= S_SWAP4;
            end

            S_SWAP4: begin
                // Conv4 output (64 ch × 24) ready; start GAP+FC
                gap_start <= 1'b1;
                state     <= S_GAPFC;
            end

            S_GAPFC: begin
                if (gap_done) state <= S_OUTPUT;
            end

            S_OUTPUT: begin
                pred_class <= gap_pred;
                valid_out  <= 1'b1;
                busy       <= 1'b0;
                state      <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

// =============================================================================
// ecg_top.v
// Top-level module - connects all submodules
//
// Weights & biases: stored in real Xilinx Block Memory Generator IP BRAMs
// (Conv1_W..Conv4_W, Conv1_B..Conv4_B, Fc_W, Fc_B), pre-initialized via
// their .coe files at build time. These are NEVER written at runtime -
// no weight write ports exist on this module, matching your existing
// .coe-programmed BRAM cells exactly as they are.
//
// Activation samples: the ECG beat IS writable at runtime via
// act_wr_en/act_wr_addr/act_wr_data, so a testbench can load a NEW
// beat before each `trigger` pulse and run many test cases in one sim,
// all using the same constant, .coe-loaded weights.
//
// External interfaces:
//   - ECG input: 187 INT8 samples written into act_bram bank A via
//     act_wr_en/addr/data BEFORE asserting trigger (only write while
//     busy == 0, to avoid colliding with conv_engine's own writes).
//   - pred_class output: valid one cycle after valid_out pulses.
//
// Target: Xilinx Artix-7 (xc7a35t or larger)
// Clock: matches your existing clk domain (see testbench CLK_PERIOD)
// =============================================================================

module ecg_top (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       trigger,
    output wire       busy,
    output wire       valid_out,
    output wire [2:0] pred_class,

    // Activation input (load ECG samples before trigger; weights are
    // fixed in BRAM via .coe and are not exposed for writing here)
    input  wire        act_wr_en,
    input  wire [14:0]  act_wr_addr,
    input  wire [7:0]   act_wr_data
);

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam P      = 8;
    localparam K_MAX  = 7;
    localparam DATA_W = 8;
    localparam ACC_W  = 32;
    localparam ADDR_W = 16;
    localparam N_CLS  = 5;
    localparam CH_GAP = 64;
    localparam T_GAP  = 24;

    // =========================================================================
    // FSM / Engine control wires
    // =========================================================================
    wire        conv_start, conv_done;
    wire [7:0]  conv_cfg_cin, conv_cfg_cout;
    wire [2:0]  conv_cfg_ksize;
    wire [1:0]  conv_cfg_stride, conv_cfg_pad;
    wire [8:0]  conv_cfg_lin, conv_cfg_lout;
    wire        gap_start, gap_done;
    wire [2:0]  gap_pred;
    wire        act_bank_sel;
    wire [1:0]  wgt_layer_sel;

    // =========================================================================
    // Activation BRAM wires
    // Port A = conv engine writes output activations, OR testbench/host
    //          writes a new ECG beat before trigger (muxed below).
    // Port B = conv engine reads input / gap_fc reads post-conv4
    // =========================================================================
    wire [14:0] act_rd_addr_conv, act_rd_addr_gap;
    wire        act_rd_en_conv,   act_rd_en_gap;
    wire [7:0]  act_rd_data;
    wire [14:0] act_wr_addr_conv;
    wire        act_wr_en_conv;
    wire [7:0]  act_wr_data_conv;

    // Read mux: GAP reads after conv done (FSM ensures no conflict)
    wire [14:0] act_bram_ra  = gap_start ? act_rd_addr_gap : act_rd_addr_conv;
    wire        act_bram_ren = gap_start ? act_rd_en_gap   : act_rd_en_conv;

    // Write mux: host (testbench) write has priority. Only assert
    // act_wr_en while busy == 0 to avoid colliding with conv_engine.
    wire        act_bram_wen  = act_wr_en | act_wr_en_conv;
    wire [14:0] act_bram_wa   = act_wr_en ? act_wr_addr : act_wr_addr_conv;
    wire [7:0]  act_bram_wdat = act_wr_en ? act_wr_data : act_wr_data_conv;

    // =========================================================================
    // Activation BRAM
    // In Block Memory Generator IP settings:
    //   - Simple Dual Port, 8-bit wide, depth 32768
    //   - No .coe needed here - contents are written at runtime
    //     (by testbench for ECG samples, by conv_engine for intermediate acts)
    // =========================================================================
    act_bram act_bram_inst (
        .clka  (clk),
        .ena   (act_bram_wen),
        .wea   (act_bram_wen),
        .addra (act_bram_wa),
        .dina  (act_bram_wdat),

        .clkb  (clk),
        .enb   (act_bram_ren),
        .addrb (act_bram_ra),
        .doutb (act_rd_data)
    );

    // =========================================================================
    // Conv Weight BRAMs - CONSTANT, .coe-initialized, unchanged from your
    // existing design. No write ports exposed; these are never touched
    // at runtime.
    // =========================================================================
    wire [15:0] wgt_rd_addr;
    wire        wgt_rd_en;
    wire [7:0]  wgt_dout_conv1, wgt_dout_conv2,
                wgt_dout_conv3, wgt_dout_conv4;

    wire [7:0] wgt_rd_data_byte;
    assign wgt_rd_data_byte = (wgt_layer_sel == 2'd0) ? wgt_dout_conv1 :
                              (wgt_layer_sel == 2'd1) ? wgt_dout_conv2 :
                              (wgt_layer_sel == 2'd2) ? wgt_dout_conv3 :
                                                        wgt_dout_conv4;

    Conv1_W conv1_w_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(8'd0),
        .clkb(clk),
        .enb  (wgt_rd_en & (wgt_layer_sel == 2'd0)),
        .addrb(wgt_rd_addr[7:0]),
        .doutb(wgt_dout_conv1)
    );
    Conv2_W conv2_w_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(8'd0),
        .clkb(clk),
        .enb  (wgt_rd_en & (wgt_layer_sel == 2'd1)),
        .addrb(wgt_rd_addr[13:0]),
        .doutb(wgt_dout_conv2)
    );
    Conv3_W conv3_w_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(8'd0),
        .clkb(clk),
        .enb  (wgt_rd_en & (wgt_layer_sel == 2'd2)),
        .addrb(wgt_rd_addr[14:0]),
        .doutb(wgt_dout_conv3)
    );
    Conv4_W conv4_w_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(8'd0),
        .clkb(clk),
        .enb  (wgt_rd_en & (wgt_layer_sel == 2'd3)),
        .addrb(wgt_rd_addr[14:0]),
        .doutb(wgt_dout_conv4)
    );

    wire [DATA_W*P*K_MAX-1:0] wgt_rd_data_wide;
    assign wgt_rd_data_wide = {{(DATA_W*P*K_MAX-DATA_W){1'b0}}, wgt_rd_data_byte};

    // =========================================================================
    // Conv Bias BRAMs - CONSTANT, .coe-initialized, unchanged.
    // =========================================================================
    wire [15:0] bias_rd_addr;
    wire        bias_rd_en;
    wire [15:0] bias_rd_data;
    wire [15:0] bias_dout_conv1, bias_dout_conv2,
                bias_dout_conv3, bias_dout_conv4;

    assign bias_rd_data = (wgt_layer_sel == 2'd0) ? bias_dout_conv1 :
                          (wgt_layer_sel == 2'd1) ? bias_dout_conv2 :
                          (wgt_layer_sel == 2'd2) ? bias_dout_conv3 :
                                                    bias_dout_conv4;

    Conv1_B conv1_b_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(16'd0),
        .clkb(clk),
        .enb  (bias_rd_en & (wgt_layer_sel == 2'd0)),
        .addrb(bias_rd_addr[4:0]),
        .doutb(bias_dout_conv1)
    );
    Conv2_B conv2_b_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(16'd0),
        .clkb(clk),
        .enb  (bias_rd_en & (wgt_layer_sel == 2'd1)),
        .addrb(bias_rd_addr[5:0]),
        .doutb(bias_dout_conv2)
    );
    Conv3_B conv3_b_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(16'd0),
        .clkb(clk),
        .enb  (bias_rd_en & (wgt_layer_sel == 2'd2)),
        .addrb(bias_rd_addr[6:0]),
        .doutb(bias_dout_conv3)
    );
    Conv4_B conv4_b_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(16'd0),.dina(16'd0),
        .clkb(clk),
        .enb  (bias_rd_en & (wgt_layer_sel == 2'd3)),
        .addrb(bias_rd_addr[5:0]),
        .doutb(bias_dout_conv4)
    );

    // =========================================================================
    // FC Weight BRAM: Fc_W (depth=320, width=8) - CONSTANT, .coe-initialized
    // =========================================================================
    wire [8:0]        fc_wgt_addr;
    wire              fc_wgt_en;
    wire signed [7:0] fc_wgt_data;

    Fc_W fc_w_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(9'd0),.dina(8'd0),
        .clkb(clk),
        .enb  (fc_wgt_en),
        .addrb(fc_wgt_addr),
        .doutb(fc_wgt_data)
    );

    // =========================================================================
    // FC Bias BRAM: Fc_B (depth=5, width=16) - CONSTANT, .coe-initialized
    // =========================================================================
    wire [2:0]         fc_bias_addr;
    wire               fc_bias_en;
    wire signed [15:0] fc_bias_data;

    Fc_B fc_b_bram (
        .clka(clk),.ena(1'b0),.wea(1'b0),.addra(3'd0),.dina(16'd0),
        .clkb(clk),
        .enb  (fc_bias_en),
        .addrb(fc_bias_addr),
        .doutb(fc_bias_data)
    );

    // =========================================================================
    // FSM Controller
    // =========================================================================
    fsm_controller fsm_inst (
        .clk            (clk),
        .rst_n          (rst_n),
        .trigger        (trigger),
        .busy           (busy),
        .valid_out      (valid_out),
        .pred_class     (pred_class),
        .conv_start     (conv_start),
        .conv_done      (conv_done),
        .conv_cfg_cin   (conv_cfg_cin),
        .conv_cfg_cout  (conv_cfg_cout),
        .conv_cfg_ksize (conv_cfg_ksize),
        .conv_cfg_stride(conv_cfg_stride),
        .conv_cfg_lin   (conv_cfg_lin),
        .conv_cfg_lout  (conv_cfg_lout),
        .conv_cfg_pad   (conv_cfg_pad),
        .gap_start      (gap_start),
        .gap_done       (gap_done),
        .gap_pred       (gap_pred),
        .act_bank_sel   (act_bank_sel),
        .wgt_layer_sel  (wgt_layer_sel)
    );

    // =========================================================================
    // Conv Engine
    // =========================================================================
    conv_engine #(
        .P(P), .K_MAX(K_MAX), .DATA_W(DATA_W),
        .ACC_W(ACC_W), .ADDR_W(ADDR_W)
    ) conv_inst (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (conv_start),
        .done           (conv_done),
        .cfg_cin        (conv_cfg_cin),
        .cfg_cout       (conv_cfg_cout),
        .cfg_ksize      (conv_cfg_ksize),
        .cfg_stride     (conv_cfg_stride),
        .cfg_lin        (conv_cfg_lin),
        .cfg_lout       (conv_cfg_lout),
        .cfg_pad        (conv_cfg_pad),
        .act_rd_addr    (act_rd_addr_conv),
        .act_rd_en      (act_rd_en_conv),
        .act_rd_data    (act_rd_data),
        .act_wr_addr    (act_wr_addr_conv),
        .act_wr_en      (act_wr_en_conv),
        .act_wr_data    (act_wr_data_conv),
        .wgt_rd_addr    (wgt_rd_addr),
        .wgt_rd_en      (wgt_rd_en),
        .wgt_rd_data    (wgt_rd_data_wide),
        .bias_rd_addr   (bias_rd_addr),
        .bias_rd_en     (bias_rd_en),
        .bias_rd_data   (bias_rd_data)
    );

    // =========================================================================
    // GAP + FC Unit
    // =========================================================================
    gap_fc_unit #(
        .CH(CH_GAP), .T(T_GAP), .N_CLS(N_CLS),
        .DATA_W(DATA_W), .ACC_W(ACC_W), .ADDR_W(ADDR_W)
    ) gap_inst (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (gap_start),
        .done         (gap_done),
        .pred_class   (gap_pred),
        .act_rd_addr  (act_rd_addr_gap),
        .act_rd_en    (act_rd_en_gap),
        .act_rd_data  (act_rd_data),
        .fc_wgt_addr  (fc_wgt_addr),
        .fc_wgt_en    (fc_wgt_en),
        .fc_wgt_data  (fc_wgt_data),
        .fc_bias_addr (fc_bias_addr),
        .fc_bias_en   (fc_bias_en),
        .fc_bias_data (fc_bias_data)
    );

endmodule