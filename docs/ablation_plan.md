# Ablation Plan

## Protocol

- Dataset: COCO train2017, validate on val2017.
- Input: 640x640 single-scale for core ablations, 640 to 960 multi-scale for final.
- Variant: Plasformers-S for component isolation, Plasformers-B for confirmation.
- Training: 300 epochs for screening, 500 epochs for final.
- Report: AP, AP50, AP75, APS, APM, APL, params, FLOPs, peak memory, T4 TensorRT FP16 FPS, ONNX export success.

## Component Isolation

| ID | Frequency path | Linear attention | Fusion gates | Neck exchange | Dynamic head | Expected observation |
|---|---|---|---|---|---|---|
| A0 | off | off | static mean | PAN-lite | static | baseline efficiency |
| A1 | on | off | static mean | PAN-lite | static | higher APS, blur robustness |
| A2 | off | on | static mean | PAN-lite | static | higher APL and occlusion robustness |
| A3 | on | on | static mean | PAN-lite | static | interaction between spectral and global context |
| A4 | on | on | adaptive | PAN-lite | static | input-adaptive specialization |
| A5 | on | on | adaptive | Plasformers neck | static | cross-scale AP gains |
| A6 | on | on | adaptive | Plasformers neck | dynamic | final head contribution |

## Frequency Branch

Experiments:

- Replace FFT path with identity.
- Replace FFT path with depthwise proxy during training.
- Use 2, 4, and 6 radial bands.
- Freeze versus learn band gains.

Metrics:

- COCO APS.
- VisDrone tiny-object AP.
- Low-light and Gaussian blur corruption AP.
- Export-mode accuracy delta.

Hypothesis: FFT gating recovers small and blurred objects by preserving discriminative high-frequency components while suppressing noise bands.

## Linear Attention

Experiments:

- Remove global path.
- Use token grids 4x4, 6x6, 8x8, 10x10.
- Compare cross-window pooled tokens to local-only depthwise convolution.

Metrics:

- APL and crowded-scene AP.
- Latency at 640 and 960.
- GPU memory during training.

Hypothesis: compressed attention gives most of the occlusion and long-range context benefit of transformers without full quadratic cost.

## Fusion Gates

Experiments:

- Static equal weighting.
- Learned global static weights.
- Input-adaptive softmax weights.
- Entropy regularization on gates.

Metrics:

- AP by object scale.
- Gate entropy by dataset domain.
- Calibration under corruptions.

Hypothesis: adaptive gates let aerial and medical domains favor frequency/local paths while autonomous scenes favor global context under occlusion.

## Neck

Experiments:

- PAN.
- BiFPN-style weighted add.
- Plasformers bidirectional neck without sparse exchange.
- Full neck with deformable resampling and sparse token exchange.

Metrics:

- APS and APM.
- Miss rate on small crowded classes.
- TensorRT latency.

Hypothesis: sparse cross-scale exchange improves small-object recall by letting P3 query high-level context without dense all-level attention.

## Dynamic Head

Experiments:

- Static decoupled head.
- Scene-conditioned dynamic depthwise head.
- Add IoU branch.
- Add auxiliary one-to-many branch.
- Add auxiliary consistency loss.

Metrics:

- AP75.
- localization error.
- confidence calibration.
- training stability from scratch.

Hypothesis: scene-conditioned kernels improve dense industrial and aerial scenes where the optimal classification/regression filters vary with scale distribution.

## Robustness Benchmarks

- COCO-C style blur, noise, brightness, contrast.
- ExDark or low-light object detection transfer.
- VisDrone and DOTA for aerial imagery.
- MVTec-style industrial defects converted to boxes.
- Medical lesion detection with CT/X-ray/ultrasound protocol.

## Deployment Benchmarks

- PyTorch eager FP32 and AMP.
- ONNX Runtime CUDA FP16.
- TensorRT FP16 and INT8 QAT.
- OpenVINO FP16.
- CoreML Neural Engine static 640 and 320.
- TPU v5e BF16 throughput with export-mode graph.
