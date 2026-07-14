// =============================================================================
// dsp48e1_wrap.v  (FIXED)
// Xilinx Artix-7 DSP48E1 primitive wrapper
//
// CHANGES vs original:
//   Explicitly tied off PCOUT, CARRYINSEL, MULTSIGNIN, RSTCTRL, which were
//   previously left unconnected (Synth 8-7071 / Synth 8-7023 warnings).
//   These have no functional effect here (no cascading, no external carry
//   select, no dedicated ctrl-signal reset domain) but leaving them
//   unconnected forces Vivado to guess and clutters the synthesis log.
//
// Computes: P_out = A * B + C  (pipelined, 3-cycle latency)
//   A : 18-bit signed  (INT8 activation, sign-extended)
//   B : 18-bit signed  (INT8 weight,     sign-extended)
//   C : 48-bit signed  (accumulator feed-in)
//   P : 48-bit signed  (result out)
//
// Pipeline latency = 3 clock cycles:
//   Cycle 1 : A/B registered (AREG=1, BREG=1)
//   Cycle 2 : M = A*B computed (MREG=1)
//   Cycle 3 : P = M + C output (PREG=1)
//
// Usage: Instantiate P copies in mac_unit.v for parallelism.
// =============================================================================

module dsp48e1_wrap (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        ce,         // clock enable (pipeline stall when low)
    input  wire signed [17:0] A,   // sign-extended INT8 activation
    input  wire signed [17:0] B,   // sign-extended INT8 weight
    input  wire signed [47:0] C,   // accumulator input (0 for first tap)
    output wire signed [47:0] P    // result: A*B + C
);

    // Artix-7 DSP48E1 supports:
    //   30-bit A port  (we use lower 18 bits)
    //   18-bit B port
    //   48-bit C port
    //   48-bit P output
    //   OPMODE = 7'b0000101 : P = M + C  (Z=C, XY=M)
    //   ALUMODE = 4'b0000   : add

    DSP48E1 #(
        // Feature Control
        .USE_MULT       ("MULTIPLY"),   // use the multiplier
        .USE_DPORT      ("FALSE"),
        .USE_SIMD       ("ONE48"),

        // Pipeline Register Control
        .ADREG          (0),
        .AREG           (1),    // 1 pipeline stage on A
        .BREG           (1),    // 1 pipeline stage on B
        .CREG           (1),    // 1 pipeline stage on C
        .DREG           (0),
        .MREG           (1),    // 1 pipeline stage on M (after multiply)
        .PREG           (1),    // 1 pipeline stage on P (output)
        .ACASCREG       (1),
        .BCASCREG       (1),

        // Reset/Clock Enable
        .IS_ALUMODE_INVERTED (4'b0),
        .IS_CARRYIN_INVERTED (1'b0),
        .IS_CLK_INVERTED     (1'b0),
        .IS_INMODE_INVERTED  (5'b0),
        .IS_OPMODE_INVERTED  (7'b0),

        // Data Widths
        .A_INPUT        ("DIRECT"),
        .B_INPUT        ("DIRECT")
    ) dsp_inst (
        // Clock and Control
        .CLK            (clk),
        .RSTA           (~rst_n),
        .RSTB           (~rst_n),
        .RSTC           (~rst_n),
        .RSTM           (~rst_n),
        .RSTP           (~rst_n),
        .RSTALLCARRYIN  (~rst_n),
        .RSTALUMODE     (~rst_n),
        .RSTINMODE      (~rst_n),
        .RSTD           (~rst_n),
        .RSTCTRL        (~rst_n),   // tie to same reset domain (was unconnected)

        .CEA1           (1'b0),   // only use single A register stage
        .CEA2           (ce),
        .CEB1           (1'b0),
        .CEB2           (ce),
        .CEC            (ce),
        .CEM            (ce),
        .CEP            (ce),
        .CEAD           (1'b0),
        .CEALUMODE      (1'b1),
        .CECTRL         (1'b1),
        .CED            (1'b0),
        .CEINMODE       (1'b1),
        .CECARRYIN      (1'b1),

        // Data Inputs
        .A              ({12'b0, A}),   // zero-pad to 30 bits
        .B              (B),            // 18 bits fits directly
        .C              (C),
        .D              (25'b0),
        .ACIN           (30'b0),
        .BCIN           (18'b0),
        .PCIN           (48'b0),
        .MULTSIGNIN     (1'b0),         // no cascaded sign-in (was unconnected)
        .CARRYCASCIN    (1'b0),
        .CARRYIN        (1'b0),
        .CARRYINSEL     (3'b000),       // select CARRYIN (was unconnected)

        // Operation Control
        // OPMODE[6:4]=000 -> Z=0 not cascade
        // OPMODE[3:2]=01  -> X,Y = M (multiplier result)
        // OPMODE[1:0]=01  -> Z = C port
        // Full: Z=C, XY=M => P = M + C
        .OPMODE         (7'b011_0101),
        .ALUMODE        (4'b0000),      // P = Z + (X : Y)
        .INMODE         (5'b0),

        // Outputs
        .P              (P),
        .PCOUT          (),             // no cascade out (was unconnected)
        .CARRYOUT       (),
        .CARRYCASCOUT   (),
        .MULTSIGNOUT    (),
        .OVERFLOW       (),
        .UNDERFLOW      (),
        .PATTERNDETECT  (),
        .PATTERNBDETECT (),
        .ACOUT          (),
        .BCOUT          ()
    );

endmodule