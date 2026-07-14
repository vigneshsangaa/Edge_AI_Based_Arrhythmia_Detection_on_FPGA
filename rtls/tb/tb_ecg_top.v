`timescale 1ns / 1ps
module tb;
    // =========================================================================
    // Clock and reset
    // =========================================================================
    reg clk   = 0;
    reg rst_n = 0;
    localparam CLK_PERIOD = 14;  // 14 ns ≈ 71.4 MHz
    always #(CLK_PERIOD/2) clk = ~clk;
    initial begin
        #(CLK_PERIOD * 5);
        rst_n = 1;
    end

    // =========================================================================
    // DUT signals
    // =========================================================================
    reg        trigger  = 0;
    wire       busy;
    wire       valid_out;
    wire [2:0] pred_class;

    // act_wr signals - used by load_ecg task to write samples into act_bram
    reg        act_wr_en   = 0;
    reg [15:0] act_wr_addr = 0;
    reg [7:0]  act_wr_data = 0;

    // =========================================================================
    // DUT instantiation
    // Weights/bias are loaded ONCE inside ecg_top from .mem files at time 0.
    // Only ECG activation samples are driven from this testbench.
    // =========================================================================
    ecg_top dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .trigger     (trigger),
        .busy        (busy),
        .valid_out   (valid_out),
        .pred_class  (pred_class),
        .act_wr_en   (act_wr_en),
        .act_wr_addr (act_wr_addr),
        .act_wr_data (act_wr_data)
    );

    // =========================================================================
    // ECG sample buffer + load task
    // Loads 187 samples into bank A (addresses 0..186).
    // IMPORTANT: only call this while busy == 0, so it never collides with
    // conv_engine writing intermediate activations into the same act_bram.
    // =========================================================================
    reg [7:0] ecg_beat [0:186];

    task load_ecg;
        integer k;
        begin
            if (busy) begin
                $display("[%0t] ERROR: load_ecg called while busy=1 - aborting load", $time);
                disable load_ecg;
            end
            for (k = 0; k < 187; k = k + 1) begin
                @(posedge clk);
                act_wr_en   <= 1'b1;
                act_wr_addr <= k[15:0];
                act_wr_data <= ecg_beat[k];
            end
            @(posedge clk);
            act_wr_en <= 1'b0;
            repeat(3) @(posedge clk);
        end
    endtask

    // =========================================================================
    // Run one test case: load a beat file, trigger inference, report result
    // =========================================================================
    integer cycle_count;

    always @(posedge clk)
        if (busy) cycle_count <= cycle_count + 1;

    task run_case;
        input [8*32-1:0] mem_file;   // filename string
        input [8*32-1:0] label;      // test case label for display
        integer expected_unused;
        begin
            cycle_count = 0;
            $display("\n[%0t] Loading %0s ...", $time, label);
            $readmemh(mem_file, ecg_beat);
            load_ecg;

            $display("[%0t] Asserting trigger...", $time);
            @(posedge clk);
            trigger <= 1'b1;
            @(posedge clk);
            trigger <= 1'b0;

            @(posedge valid_out);
            @(posedge clk);
            $display("[%0t] === Inference complete (%0s) ===", $time, label);
            $display("Predicted class : %0d", pred_class);
            case (pred_class)
                3'd0: $display("  -> Normal (N)");
                3'd1: $display("  -> Supraventricular (S)");
                3'd2: $display("  -> Ventricular (V)");
                3'd3: $display("  -> Fusion (F)");
                3'd4: $display("  -> Unknown (Q)");
                default: $display("  -> INVALID CLASS %0d", pred_class);
            endcase
            $display("Inference cycles : %0d  (%.3f us at 71.4 MHz)",
                     cycle_count, cycle_count * 0.014);
        end
    endtask

    // =========================================================================
    // Test sequence - 5 beats, one per class, all in a single simulation run
    // =========================================================================
    initial begin
        $dumpfile("tb.vcd");
        $dumpvars(0, tb);

        @(posedge rst_n);
        repeat(10) @(posedge clk);

        run_case("class0_sample0.mem", "Beat 1 - Normal (class 0)");
        run_case("class1_sample0.mem", "Beat 2 - Supraventricular (class 1)");
        run_case("class2_sample0.mem", "Beat 3 - Ventricular (class 2)");
        run_case("class3_sample0.mem", "Beat 4 - Fusion (class 3)");
        run_case("class4_sample0.mem", "Beat 5 - Unknown (class 4)");

        #(CLK_PERIOD * 20);
        $display("\nAll test cases complete.");
        $finish;
    end

    // =========================================================================
    // Timeout watchdog - 500,000 cycles max
    // =========================================================================
    initial begin
        #(CLK_PERIOD * 500000);
        $display("TIMEOUT: inference did not complete within 500,000 cycles");
        $finish;
    end
endmodule