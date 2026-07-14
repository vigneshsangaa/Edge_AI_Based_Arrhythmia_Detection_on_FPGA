# Edge-AI Based Arrhythmia Detection on FPGA

A compact 1D-CNN trained in PyTorch on the **MIT-BIH Arrhythmia Dataset**, quantized to INT8, and deployed as a fully custom Verilog RTL accelerator on a **Xilinx Artix-7 (xc7a35t-ftg256)** FPGA. The design classifies a single 187-sample ECG heartbeat into one of 5 arrhythmia classes (N, S, V, F, Q) end-to-end in hardware — from raw ECG samples in, to a 3-bit predicted class out.

> Training happens once, offline, in PyTorch. Everything downstream — quantization, weight export, and inference — runs on the FPGA fabric with no soft-core CPU involved.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Model & Dataset](#model--dataset)
- [Hardware Architecture](#hardware-architecture)
- [Fixed-Point / Quantization Scheme](#fixed-point--quantization-scheme)
- [How to Reproduce](#how-to-reproduce)
  - [1. Train the model (software)](#1-train-the-model-software)
  - [2. Export weights for FPGA](#2-export-weights-for-fpga)
  - [3. Build & simulate the RTL](#3-build--simulate-the-rtl)
  - [4. Synthesize & implement in Vivado](#4-synthesize--implement-in-vivado)
- [Results](#results)
  - [Model Accuracy](#model-accuracy)
  - [FPGA Resource Utilization](#fpga-resource-utilization)
  - [Timing](#timing)
  - [Power](#power)
- [Known Issues Fixed During Development](#known-issues-fixed-during-development)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

| | |
|---|---|
| **Task** | 5-class ECG beat classification (N, S, V, F, Q) |
| **Dataset** | MIT-BIH Arrhythmia Dataset (`mitbih_train.csv`, `mitbih_test.csv`) |
| **Training framework** | PyTorch |
| **Deployment format** | INT8 fixed-point, BatchNorm-fused, `$readmemh`-compatible `.mem` / Vivado `.coe` |
| **Target FPGA** | Xilinx Artix-7, `xc7a35t-ftg256-1` (7-series, PRODUCTION speed grade) |
| **Toolchain** | Vivado 2025.2, PyTorch, ONNX / ONNX Runtime |
| **Export formats** | `.pth` (PyTorch), `.onnx` (interchange), `.mem` + `.coe` (Verilog BRAM init) |

### Class labels

| Code | Class | Clinical meaning |
|---|---|---|
| N | Normal | Normal sinus beat |
| S | Supraventricular | Ectopic beat originating above the ventricles |
| V | Ventricular | Ectopic beat originating in the ventricles |
| F | Fusion | Fusion of a normal and a ventricular beat |
| Q | Unknown | Unclassifiable / paced beat |

The MIT-BIH training set is heavily imbalanced (N beats dominate; S and F are rare), which is addressed with a weighted sampler and a class-weighted focal loss during training (see [Model & Dataset](#model--dataset)).

---

## Repository Structure

```
Edge_AI_Based_Arrhythmia_Detection_on_FPGA/
├── ml/                                   # Software: training, quantization, export
│   ├── ECG_Arrhythmia_Detection.ipynb    # End-to-end training/export notebook
│   ├── Dataset/
│   │   ├── mitbih_train.csv
│   │   └── mitbih_test.csv
│   ├── ecg_arrhythmia_model.pth          # Trained PyTorch weights (float32)
│   ├── ecg_arrhythmia_model.onnx         # Exported ONNX graph
│   ├── ecg_arrhythmia_model.onnx.data    # ONNX external weight data
│   ├── scaler_mean.npy                   # StandardScaler mean (FPGA input preprocessing)
│   ├── scaler_std.npy                    # StandardScaler std  (FPGA input preprocessing)
│   ├── class_imbalance.png               # Dataset class distribution plot
│   ├── imbalance_analysis.png            # F1 vs. support plot
│   ├── ecg_samples.png                   # Sample ECG waveform per class
│   ├── evaluation_results.png            # Confusion matrix + training curves
│   ├── fpga_coe_files/                   # INT8 weights/biases as Xilinx .coe (BRAM IP init)
│   │   ├── conv1_weight.coe / conv1_bias.coe
│   │   ├── conv2_weight.coe / conv2_bias.coe
│   │   ├── conv3_weight.coe / conv3_bias.coe
│   │   ├── conv4_weight.coe / conv4_bias.coe
│   │   └── fc_weight.coe    / fc_bias.coe
│   ├── fpga_mem_files/                   # Same weights as $readmemh-compatible .mem + .txt
│   │   ├── conv1_weight.mem / conv1_bias.mem  (+ .txt float debug dumps)
│   │   ├── conv2_weight.mem / conv2_bias.mem
│   │   ├── conv3_weight.mem / conv3_bias.mem
│   │   ├── conv4_weight.mem / conv4_bias.mem
│   │   └── fc_weight.mem    / fc_bias.mem
│   └── sample_mem_files/                 # Pre-quantized single-beat test vectors for simulation
│       ├── class0_sample0.mem … class0_sample4.mem   (Normal)
│       ├── class1_sample0.mem … class1_sample4.mem   (Supraventricular)
│       ├── class2_sample0.mem … class2_sample4.mem   (Ventricular)
│       ├── class3_sample0.mem … class3_sample4.mem   (Fusion)
│       ├── class4_sample0.mem … class4_sample4.mem   (Unknown)
│       └── 5_classes_samples.mem                     (bundle, one per class)
│
└── rtls/                                 # Hardware: RTL, testbench, synthesis reports
    ├── codes/
    │   ├── ecg_top.v          # Top-level module + fsm_controller (layer sequencing)
    │   ├── conv_engine.v      # Reusable 1D convolution engine (shared across Conv1..4)
    │   ├── mac_unit.v         # Parallel P-wide multiply-accumulate + ReLU/saturation
    │   ├── dsp48e1_wrap.v     # DSP48E1 primitive wrapper (A*B+C, 3-cycle pipeline)
    │   ├── line_buffer.v      # Shift-register tap extractor for convolution windows
    │   └── gap_fc_unit.v      # Global Average Pool -> Fully Connected -> Argmax
    ├── tb/
    │   └── tb_ecg_top.v       # Self-checking testbench, one beat per class
    ├── reports/                          # Vivado-generated reports (see Results)
    │   ├── ecg_top_utilization_synth.rpt
    │   ├── ecg_top_utilization_placed.rpt
    │   ├── ecg_top_control_sets_placed.rpt
    │   ├── ecg_top_io_placed.rpt
    │   ├── ecg_top_clock_utilization_routed.rpt
    │   ├── ecg_top_timing_summary_routed.rpt
    │   ├── ecg_top_bus_skew_routed.rpt
    │   ├── ecg_top_power_routed.rpt
    │   ├── ecg_top_route_status.rpt
    │   ├── ecg_top_drc_opted.rpt
    │   ├── ecg_top_drc_routed.rpt
    │   └── ecg_top_methodology_drc_routed.rpt
    └── output_images/
        ├── Synthesis.png     # Elaborated/synthesized schematic screenshot
        └── Waveform.png      # Simulation waveform screenshot
```

---

## Model & Dataset

The model is a compact, FPGA-optimized 1D-CNN, deliberately shaped so every layer's parallelism, kernel size, and channel count maps cleanly onto a small number of shared DSP48E1 slices.

**Input:** a single ECG beat, 187 samples, `float32`, standardized via `scaler_mean.npy` / `scaler_std.npy` before quantization.

| Layer | Op | In → Out shape | Kernel | Stride | Notes |
|---|---|---|---|---|---|
| Conv1 | Conv1D + BN + ReLU | (1, 187) → (32, 94) | 7 | 2 | padding = 3 |
| Conv2 | Conv1D + BN + ReLU | (32, 94) → (64, 47) | 5 | 2 | padding = 2 |
| Conv3 | Conv1D + BN + ReLU | (64, 47) → (128, 24) | 3 | 2 | padding = 1 |
| Conv4 | Conv1D + BN + ReLU (bottleneck) | (128, 24) → (64, 24) | 3 | 1 | padding = 1, halves FC input width |
| GAP | Global Average Pool | (64, 24) → (64,) | – | – | approximated on FPGA as `(sum + 12) >> 5` |
| FC | Linear | (64,) → (5,) | – | – | 5 output logits |
| Argmax | – | (5,) → class id | – | – | combinational compare tree in hardware |

**Training setup:**
- Loss: class-weighted **focal loss** (γ = 2.0), with per-class weights `[N:1.0, S:2.0, V:1.0, F:2.0, Q:1.0]` to counter the ~113:1 majority:minority class imbalance in MIT-BIH.
- Sampler: `WeightedRandomSampler` so every mini-batch sees a more balanced class mix.
- Augmentation: per-sample additive Gaussian noise, random circular time shift, random amplitude scaling.
- Optimizer: Adam, cosine-annealed LR schedule, gradient clipping, smoothed early stopping.
- BatchNorm is **fused into the preceding Conv weights/bias** before export, so the FPGA never has to compute a BatchNorm — Conv+ReLU is the only compute primitive it needs.

Full training code, plots, and evaluation are in [`ml/ECG_Arrhythmia_Detection.ipynb`](ml/ECG_Arrhythmia_Detection.ipynb).

---

## Hardware Architecture

```
                       ┌──────────────────────────┐
   ECG beat (187×INT8) │        act_bram           │  Conv1..4 intermediate
   loaded via           │  (ping-pong banks A/B)    │  activations (same BRAM,
   act_wr_en/addr/data  └───────────┬──────────────┘  bank-swapped each layer)
                                    │
                         ┌──────────▼──────────┐
                         │   fsm_controller      │  sequences Conv1→Conv2→Conv3→
                         │   (in ecg_top.v)      │  Conv4→GAP+FC→output, one shot
                         └──────┬───────┬───────┘  per `trigger` pulse
                                │       │
                  ┌─────────────▼───┐ ┌─▼──────────────┐
                  │   conv_engine     │ │  gap_fc_unit    │
                  │  (reused for all  │ │ (GAP → FC →     │
                  │  4 conv layers)   │ │  Argmax)        │
                  └─────────┬────────┘ └────────┬────────┘
                            │                    │
                     ┌──────▼──────┐             │
                     │  mac_unit    │             │
                     │  (P=8 lanes) │             │
                     └──────┬──────┘             │
                            │                     │
                  ┌─────────▼─────────┐           │
                  │  P × dsp48e1_wrap  │           │
                  │  (DSP48E1 A*B+C,   │           │
                  │   3-cycle pipeline)│           │
                  └────────────────────┘           │
                                                    ▼
                                          pred_class[2:0] + valid_out
```

- **`ecg_top.v`** — top-level integration. Instantiates the activation BRAM (runtime-writable), the constant `.coe`-initialized weight/bias BRAMs (`Conv1_W`…`Conv4_W`, `Conv1_B`…`Conv4_B`, `Fc_W`, `Fc_B`), the `fsm_controller`, `conv_engine`, and `gap_fc_unit`. Also contains `fsm_controller`, the top-level FSM that walks through Conv1 → Conv2 → Conv3 → Conv4 → GAP/FC → output for every `trigger` pulse, reconfiguring `conv_engine`'s per-layer parameters (`cin`, `cout`, `ksize`, `stride`, `lin`, `lout`, `pad`) at each stage.
- **`conv_engine.v`** — a single, reusable 1D convolution engine, time-multiplexed across all 4 conv layers. Streams activations through a `line_buffer`, loads weights per (output-channel-group, input-channel) pair, drives `mac_unit`, and performs bias-load + write-back. `P = 8` parallel output channels per cycle; `Cout` is padded to a multiple of `P` where needed.
- **`mac_unit.v`** — `P` parallel DSP48E1 lanes, each accumulating `act_tap * weight` across kernel taps and input channels into a 32-bit accumulator, then applying folded-BN bias + ReLU + INT8 saturation.
- **`dsp48e1_wrap.v`** — thin wrapper around the Xilinx `DSP48E1` primitive computing `P = A*B + C` with a 3-cycle pipeline (`AREG=BREG=1`, `MREG=1`, `PREG=1`). All optional control ports (`PCOUT`, `CARRYINSEL`, `MULTSIGNIN`, `RSTCTRL`) are explicitly tied off to eliminate synthesis warnings.
- **`line_buffer.v`** — parametric shift-register tap extractor (max kernel depth `K=7`), maps to small distributed-RAM/SRL logic, no BRAM needed.
- **`gap_fc_unit.v`** — after Conv4, performs Global Average Pooling (64 channels × 24 samples → 64 values, via `(sum + 12) >> 5` rounding-shift approximation of ÷24), streams in the 320 FC weights and 5 biases from BRAM, computes 5 output logits, and finds the arg-max class combinationally.
- **`tb/tb_ecg_top.v`** — self-checking testbench that loads one ECG beat per class from `ml/fpga_mem_files/sample_mem_files/`, pulses `trigger`, waits for `valid_out`, and reports the predicted class + inference cycle count for all 5 classes in a single simulation run.

---

## Fixed-Point / Quantization Scheme

- **Weights & activations:** INT8, two's-complement.
- **Bias:** INT16 (loaded 1 value/cycle from a dedicated bias BRAM via a `S_BIAS_LOAD` FSM state).
- **Accumulator:** INT32 (`ACC_W = 32`), sign-extended into a 33-bit adder alongside the INT16 bias before ReLU + saturation to `[0, 127]`.
- **BatchNorm fusion:** `w_fused = w * (γ / sqrt(var + eps))`, `b_fused = β − mean * (γ / sqrt(var + eps))`, computed once in Python and exported — the FPGA only ever sees Conv+bias+ReLU.
- **GAP division:** true ÷24 is approximated on-chip as `(sum + 12) >> 5` (i.e. ÷32 with rounding), trading a small amount of accuracy for a divider-free implementation. A note in `gap_fc_unit.v` documents the option to swap in a real divider if higher fidelity is needed.
- **Weight/bias export:** every layer's fused INT8 weights and INT16 biases are written out twice — as Xilinx `.coe` (for Block Memory Generator IP initialization, `ml/fpga_coe_files/`) and as plain hex `.mem` (for `$readmemh` in simulation/synthesis, `ml/fpga_mem_files/`), plus a float `.txt` dump of each for debugging/cross-checking against the PyTorch model.

---

## How to Reproduce

### 1. Train the model (software)

```bash
cd ml
# open and run top-to-bottom in Jupyter / Colab
jupyter notebook ECG_Arrhythmia_Detection.ipynb
```

Requirements: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `onnx`, `onnxruntime`. Place `mitbih_train.csv` and `mitbih_test.csv` (MIT-BIH Arrhythmia Dataset, Kaggle) under `ml/Dataset/`.

Running the notebook end-to-end will:
1. Load & explore the dataset, visualize class imbalance.
2. Train the FPGA-optimized 1D-CNN with focal loss + weighted sampling.
3. Evaluate on the held-out test set (confusion matrix, per-class precision/recall/F1).
4. Fuse BatchNorm into Conv weights.
5. Export `.pth`, `.onnx`, and INT8 `.coe` / `.mem` weight files under `ml/fpga_coe_files/` and `ml/fpga_mem_files/`.

### 2. Export weights for FPGA

This step is included in the notebook (Sections 23–24: *INT8 Quantization* and *Export `.mem` Weight Files for Verilog*), and produces:
- `ml/fpga_coe_files/*.coe` — for initializing Vivado Block Memory Generator IP (`Conv1_W`, `Conv1_B`, …, `Fc_W`, `Fc_B`) at bitstream build time.
- `ml/fpga_mem_files/*.mem` — for `$readmemh` in simulation.
- `ml/sample_mem_files/*.mem` — quantized single-beat test vectors (5 samples × 5 classes) for driving the testbench.

### 3. Build & simulate the RTL

1. Create a Vivado project targeting `xc7a35t-ftg256-1` (or a compatible Artix-7 part).
2. Add all sources from `rtls/codes/` (`ecg_top.v`, `conv_engine.v`, `mac_unit.v`, `dsp48e1_wrap.v`, `line_buffer.v`, `gap_fc_unit.v`).
3. Add the testbench `rtls/tb/tb_ecg_top.v` as a simulation-only source.
4. Generate the required Block Memory Generator IP cores and initialize each with its matching `.coe` from `ml/fpga_coe_files/`:
   - `act_bram` — Simple Dual Port, 8-bit wide, depth 32768, **no** `.coe` (written at runtime by the testbench/host).
   - `Conv1_W` … `Conv4_W`, `Conv1_B` … `Conv4_B`, `Fc_W`, `Fc_B` — constant, `.coe`-initialized, read-only at runtime.
5. Run behavioral simulation. The testbench (`tb_ecg_top.v`) automatically:
   - Loads one 187-sample beat per class from `ml/sample_mem_files/class{0..4}_sample0.mem`,
   - Pulses `trigger`,
   - Waits for `valid_out`,
   - Prints the predicted class name and the number of clock cycles the inference took, for all 5 classes back-to-back.
6. Inspect `rtls/output_images/Waveform.png` for a reference waveform, or view `tb.vcd` in your simulator's waveform viewer.

### 4. Synthesize & implement in Vivado

1. Run **Synthesis** → **Implementation** → **Generate Bitstream** in Vivado on the same project.
2. Reports equivalent to those under `rtls/reports/` will be regenerated (utilization, timing, power, DRC, methodology, I/O, control sets, clock utilization, bus skew, route status).
3. `rtls/output_images/Synthesis.png` shows the elaborated/synthesized schematic for reference.

---

## Results

### Model Accuracy

| Metric | Value |
|---|---|
| Overall test accuracy | ~82% (see `ml/evaluation_results.png` for the exact run's confusion matrix) |
| INT8 vs. float32 accuracy drop | measured in-notebook (Section 25 constraint summary), target < 1% |
| Model size (INT8, fused BN) | ~59 KB total weights/biases across all layers |

S and F are the hardest classes to classify — they have the fewest training examples and their waveforms morphologically overlap with N and V respectively (see `ml/imbalance_analysis.png` for the F1-vs-support breakdown, and `ml/evaluation_results.png` for the full confusion matrix and training curves).

### FPGA Resource Utilization

**Post-synthesis** (`rtls/reports/ecg_top_utilization_synth.rpt`) vs. **post-place** (`rtls/reports/ecg_top_utilization_placed.rpt`), target `xc7a35t-ftg256-1`:

| Resource | Used (placed) | Available | Utilization |
|---|---|---|---|
| Slice LUTs | 3,988 | 20,800 | 19.17% |
| Slice Registers | 4,649 | 41,600 | 11.18% |
| F7 Muxes | 445 | 16,300 | 2.73% |
| F8 Muxes | 162 | 8,150 | 1.99% |
| Block RAM Tiles (RAMB36+RAMB18) | 16.5 | 50 | 33.00% |
| DSP48E1 | 9 | 90 | 10.00% |
| Bonded IOBs | 32 | 170 | 18.82% |
| BUFGCTRL | 1 | 32 | 3.13% |

Only **9 DSP48E1 slices** and **~33% of on-chip Block RAM** are used, leaving substantial headroom on a small Artix-7 (`xc7a35t`) for scaling parallelism (`P`) or adding a bigger model.

### Timing

From `rtls/reports/ecg_top_timing_summary_routed.rpt`, post-route, all corners:

| Metric | Value |
|---|---|
| Target clock period | 14.000 ns (**71.43 MHz**) |
| Worst Negative Slack (WNS), setup | **+0.086 ns** (MET) |
| Worst Hold Slack (WHS) | **+0.163 ns** (MET) |
| Worst Pulse Width Slack (WPWS) | +6.500 ns (MET) |
| Failing endpoints (setup / hold / pulse width) | **0 / 0 / 0** |

All user-specified timing constraints are met at 71.43 MHz with a critical path through the FC accumulator carry chain in `gap_fc_unit` (19 logic levels: `CARRY4`/`LUT`/`MUXF7`/`MUXF8` chain). The `rtls/reports/ecg_top_bus_skew_routed.rpt` reports no bus-skew constraints defined, and `ecg_top_clock_utilization_routed.rpt` confirms a single clean global clock domain (`sys_clk`, 1 `BUFGCTRL`, no MMCM/PLL required).

### Power

From `rtls/reports/ecg_top_power_routed.rpt` (post-route, typical process, 25°C ambient):

| Metric | Value |
|---|---|
| **Total On-Chip Power** | **0.111 W** |
| Dynamic Power | 0.039 W |
| Static (Device) Power | 0.073 W |
| Clocks | 0.009 W |
| Slice Logic | 0.006 W |
| Signals | 0.016 W |
| Block RAM | 0.001 W |
| DSPs | 0.002 W |
| I/O | 0.003 W |
| Junction Temperature | 25.5 °C |
| Effective θJA | 4.9 °C/W |

Confidence level is reported as **Low** by Vivado, since I/O and internal switching activity were not driven by a full post-implementation simulation activity file (`.saif`) — the reported numbers are Vivado's vectorless estimate. For a more accurate power figure, back-annotate a real simulation activity file from `tb_ecg_top.v` before re-running `report_power`.

---

## Known Issues Fixed During Development

These were found and corrected while working through Vivado's synthesis warnings and the training notebook — documented here for transparency and as a changelog:

| # | Area | Issue | Fix |
|---|---|---|---|
| 1 | `conv_engine.v` | **Critical:** `mac_bias` was declared but never driven, and `bias_rd_data` was never consumed — every conv layer silently ran with bias = 0, corrupting inference accuracy without any synthesis error. | Added an `S_BIAS_LOAD` FSM state that sequentially reads `P` INT16 biases (1-cycle BRAM latency) and packs them into `mac_bias` before triggering the bias-add/ReLU pipeline. |
| 2 | `ecg_top.v` / `dsp48e1_wrap.v` | BRAM data/address width mismatches and unconnected DSP48E1 control ports (`PCOUT`, `CARRYINSEL`, `MULTSIGNIN`, `RSTCTRL`) were flagged by DRC (`REQP-1839`/`REQP-1840`, `AVAL-4`). | Explicitly tied off unused DSP48E1 ports to the same reset domain / constant values used elsewhere in the design. |
| 3 | `conv_engine.v` | Dead sequential element `wr_base` was assigned but never read (write address computed directly from `cg_cnt`/`t_pos`), optimized away by synthesis and cluttering the report. | Removed for a clean synthesis report. |
| 4 | Training notebook — Focal Loss | `class_loss_weights` were computed and printed as if active, but never actually passed into `FocalLoss` (`alpha` stayed `None`) — the imbalance weighting had zero real effect on training. | Wired the computed weights into `alpha` so `F.cross_entropy(..., weight=self.alpha, ...)` actually applies them. |
| 5 | Training notebook — Augmentation | `augment()` drew a single random shift/scale/noise draw and applied it identically to every sample in the batch. | Rewrote to draw independent shift/scale per sample in the batch. |
| 6 | Training notebook — Training loop | Positional argument mismatch: caller passed `(model, train_loader, val_loader, ...)` into a function whose parameters were named `(model, val_loader, test_loader, ...)` — misleading names, one bug away from leaking the test set into early-stopping decisions. | Renamed parameters to match what is actually passed; validation now unambiguously validates on the held-out validation split. |
| 7 | Training notebook — BN fusion logging | Log line hard-coded the FC layer's fused shape as `(5, 32)`; the actual layer is `nn.Linear(64, num_classes)`, i.e. `(5, 64)`. | Fixed to report the real tensor shape. |
| 8 | Training notebook — FPGA constraint check | Accuracy check used `0.9 > overall_acc > 0.8`, so a model scoring **above 90%** — better than the 80% target — would be reported as `FAIL`. | Replaced with a simple lower-bound check (`overall_acc > 0.8`). |
| 9 | Training notebook — Netron visualization | Original cell launched a blocking `netron` server directly in a notebook cell, hanging any top-to-bottom re-run (including in CI/GitHub). | Made Netron viewing opt-in (commented-out call); default behavior points to opening the exported `.onnx` at [netron.app](https://netron.app). |

---

## Conclusion

This project demonstrates a full edge-AI pipeline — from a class-imbalanced medical time-series dataset, through a quantization-aware compact CNN, down to a hand-written, resource-efficient Verilog accelerator — running entirely on a low-cost Artix-7 FPGA with no soft-core CPU in the loop.

- **Accuracy:** the INT8, BatchNorm-fused model reaches ~82% overall test accuracy on held-out MIT-BIH data, with the expected recall trade-off on the rare S/F classes documented and analyzed rather than hidden by aggregate accuracy alone.
- **Area:** the accelerator is extremely lightweight — **19.17% LUTs, 11.18% registers, 10% DSP48E1s, 33% BRAM** on an `xc7a35t`, the smallest device in the Artix-7 family, leaving significant headroom for scaling.
- **Timing:** all setup/hold/pulse-width constraints are met with positive slack at a 71.43 MHz clock (14 ns period), on a single clean global clock domain with no MMCM/PLL.
- **Power:** total on-chip power is estimated at **0.111 W** (39 mW dynamic + 73 mW static), well within the always-on power budget for a wearable or bedside ECG monitoring device — though this figure should be re-validated with a real switching-activity file for production sign-off.

Together, these results support the project's core premise: a 1D-CNN arrhythmia classifier can be quantized and hand-mapped to a small, cheap FPGA with real design margin left on area, timing, and power — making it a realistic candidate for battery-powered, always-on edge ECG monitoring.

---

## Future Work

- Push the reported INT8-vs-float32 accuracy delta down further with quantization-aware training (QAT) instead of post-training quantization.
- Replace the GAP `>>5` rounding-shift approximation with an exact ÷24 divider (or retrain with a pooling window size that is a clean power of two) to close the small accuracy gap it introduces.
- Back-annotate a real `.saif` switching-activity file from `tb_ecg_top.v` for a high-confidence, non-vectorless power estimate.
- Address the outstanding `NSTD-1`/`UCIO-1` DRC warnings (unconstrained I/O standards/locations on `act_wr_*` ports) with a proper board-level XDC before targeting real silicon.
- Explore increasing `P` (parallel MAC lanes) to trade the currently-unused DSP/BRAM headroom for lower inference latency.
- Add unit tests around `fuse_bn()` and the INT8 quantization helper in the notebook (e.g. comparing fused conv+BN output against the unfused floating-point model on held-out samples) so future edits can't silently break the FPGA export path the way bug #1 did.

---

## License

