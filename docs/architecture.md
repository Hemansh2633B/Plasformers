# Plasformers Architecture

## Name

**Plasformers** means **Progressive Local-Attentive Spectral Fusion with Object-query Routing for Multi-scale Efficient Real-time Sensing**. The name is deliberately lowercase in code, but the acronym describes the model contract:

- Progressive: five hierarchical stages with continuous fusion.
- Local-Attentive: convolutional inductive bias plus compressed attention.
- Spectral: FFT-domain frequency gating is part of the backbone, not a plug-in.
- Fusion: every block learns sample-specific branch weights.
- Object-query Routing: the head uses a scene descriptor to condition prediction kernels.
- Multi-scale Efficient Real-time Sensing: the model is built for P3-P7 detection and export.

## Macro Pipeline

```text
Input
  -> Adaptive Hybrid Stem
  -> Tri-Path Hierarchical Backbone
  -> Cross-Scale Global Fusion Neck
  -> Dynamic Query-Guided Detection Head
  -> Unified Multi-Task Output
```

## Adaptive Hybrid Stem

For input `X in R^{B x 3 x H x W}`, the stem returns `S0 in R^{B x C0 x ceil(H/4) x ceil(W/4)}`.

| Layer | Kernel | Stride | Output for 640x640, base | Norm | Activation |
|---|---:|---:|---:|---|---|
| Learnable frequency preprocessor | DW 3x3 + PW 1x1 | 1 | Bx3x640x640 | none | gated residual |
| Anti-aliased conv 1 | blur 3x3 + conv 3x3 | 2 | Bx24x320x320 | BN | SiLU |
| Detail refinement | DW 3x3 + PW 1x1 | 1 | Bx24x320x320 | BN | SiLU |
| Anti-aliased conv 2 | blur 3x3 + conv 3x3 | 2 | Bx48x160x160 | BN | SiLU |
| Detail bypass | AvgPool 4x4 + PW 1x1 | 4 | Bx48x160x160 | BN | SiLU |
| Fusion projection | 1x1 | 1 | Bx48x160x160 | BN | SiLU |

Initialization uses Kaiming normal for convolutions, unit BatchNorm scale, zero BatchNorm bias, and low objectness/class priors in the head. The detail bypass preserves high-frequency spatial evidence that would otherwise be attenuated by downsampling.

## Tri-Path Backbone

Every block has three parallel paths.

### Local Path

The local path splits channels into bypass and routed partitions. The routed partition uses pointwise expansion, a dynamic mixture of depthwise kernels, channel recalibration, and pointwise compression:

`F_local = X + phi([X_bypass, R(X_route)])`

where `R` is a depthwise kernel mixture:

`R(X) = BN(SiLU(sum_i pi_i(X) * DWConv_{k_i}(X)))`

The mixture weights `pi_i` are produced from global average pooled context.

### Global Path

The global path uses query-key compression. Full-resolution queries attend to pooled tokens:

`Q = W_q X`

`K,V = W_k P(X), W_v P(X)`

`F_global = X + W_o softmax(Q K^T / sqrt(d)) V + PE_dw(X)`

`P` is adaptive average pooling to a small token grid. This gives cross-window communication without quadratic full-image attention.

### Frequency Path

The frequency path uses FFT decomposition in training/eager mode:

`Z = FFT2(X)`

`M_c(u,v) = sum_b softmax(a_{c,b}) (1 + tanh(g_{c,b})) B_b(u,v)`

`F_freq = X + SE(IFFT2(Z * (1 + M)))`

`B_b` are radial frequency bases. Export mode replaces the FFT with a depthwise separable spectral surrogate for ONNX, TensorRT, CoreML, OpenVINO, and TPU-friendly graphs.

### Input-Adaptive Fusion

The three paths are fused with sample-specific simplex weights:

`[alpha, beta, gamma] = softmax(MLP(GAP(X)))`

`F_out = alpha F_local + beta F_global + gamma F_freq`

`alpha + beta + gamma = 1`.

## Stage Specification

Base variant at 640x640:

| Stage | Role | Stride | Tensor | Channels | Blocks | Heads | Token grid | Local kernel | Est. FLOPs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | high-resolution shallow features | 4 | 160x160 | 48 | 2 | 1 | 8x8 | 5 | 7.8G |
| S2 | low-level semantics | 8 | 80x80 | 96 | 2 | 2 | 8x8 | 5 | 8.9G |
| S3 | mid-level abstraction | 16 | 40x40 | 192 | 4 | 4 | 8x8 | 7 | 15.6G |
| S4 | deep semantic encoding | 32 | 20x20 | 384 | 3 | 8 | 6x6 | 9 | 12.4G |
| S5 | context aggregation | 64 | 10x10 | 576 | 2 | 8 | 4x4 | 11 | 5.3G |

The values above include the intended operator envelope. Exact FLOPs depend on export mode and image shape.

## Cross-Scale Global Fusion Neck

The neck consumes S1-S5 and emits P3-P7.

| Level | Source | Stride | Base tensor |
|---|---|---:|---:|
| P3 | S2 plus downsampled S1 | 8 | Bx192x80x80 |
| P4 | S3 | 16 | Bx192x40x40 |
| P5 | S4 | 32 | Bx192x20x20 |
| P6 | S5 | 64 | Bx192x10x10 |
| P7 | downsampled P6 | 128 | Bx192x5x5 |

It improves over fixed PAN/BiFPN-style routing through four mechanisms:

- Bidirectional weighted routing with positive normalized coefficients.
- Continuous-offset cross-scale aggregation through `grid_sample` during training.
- Export fallback to bilinear resampling for accelerator compatibility.
- Sparse level-token exchange, where descriptors from P3-P7 form a learned routing matrix.

The sparse exchange is:

`D_l = GAP(P_l)`

`A_l = softmax(W_r D_l / tau)`

`C_l = sum_j A_{l,j} D_j`

`P_l' = P_l * (1 + sigmoid(W_c C_l))`

## Dynamic Query-Guided Detection Head

The head is anchor-free and decoupled. A scene context vector conditions dynamic depthwise kernels for each pyramid level:

`q_scene = MLP([GAP(P3), ..., GAP(P7)])`

`H_l = P_l + sum_i softmax(W_g q_scene)_i DWConv_i(P_l)`

Each level then branches into:

- classification logits: `C` channels
- box distributions: `4 * (reg_max + 1)` channels
- objectness: `1` channel
- IoU refinement: `1` channel
- optional mask coefficients
- optional keypoints

The auxiliary branch is used only during training for one-to-many assignment and consistency regularization.

## Training

Recommended losses:

- Varifocal Loss for IoU-aware classification.
- Distribution Focal Loss for distance distributions.
- SIoU or CIoU for final box geometry.
- Auxiliary consistency loss between one-to-many and one-to-one branches.

Recommended schedule:

- 500 COCO epochs for scratch training.
- AdamW, cosine schedule, 5 epoch warmup.
- EMA with decay 0.9999.
- AMP with FP16 or BF16.
- Dynamic image scaling from 512 to 960.
- Mosaic, Copy-Paste, MixUp, HSV, affine transforms.
- Disable mosaic for the final 15 epochs.

## Deployment

Use `model.set_export_mode(True)` before export. This changes:

- FFT spectral path to depthwise separable spectral proxy.
- Deformable resampling to bilinear interpolation.
- Dictionary output to tuple output via `ExportWrapper`.

Targets:

- TensorRT: ONNX opset 17, FP16 or INT8 QAT.
- ONNX Runtime: dynamic batch and spatial axes.
- CoreML: static-size export recommended.
- OpenVINO: static or dynamic ONNX.
- TPU v5e: BF16 training and XLA-compatible export-mode graph.
- Edge TPU: use nano/small with INT8 QAT and static input.

## Efficiency Optimizations

Plasformers is designed so the research path and deployment path can be separated cleanly:

| Optimization | Mechanism | Target |
|---|---|---|
| Kernel fusion | `fuse_model_for_inference` folds Conv-BN blocks before export | TensorRT, ONNX Runtime, OpenVINO |
| Reparameterization | FFT and deformable training paths switch to static convolution/interpolation surrogates in export mode | CoreML, TensorRT, TPU |
| Quantization-aware training | `prepare_qat` attaches torch.ao QAT observers | Edge TPU, OpenVINO INT8, mobile NPU |
| Structured pruning | `collect_prunable_convs` identifies pointwise/group-free conv layers | Nano and Small variants |
| Sparse activation support | activation sparsity profiler supports post-training analysis | TensorRT sparse kernels, NPU compilers |
| Mixed precision | AMP config and training helper support FP16/BF16 | GPUs and TPU v5e |

## Scaling Family

| Variant | Params target | FLOPs target 640 | T4 TRT FP16 FPS target | COCO AP target | Primary use |
|---|---:|---:|---:|---:|---|
| Nano | 3.9M | 8.7G | 520 | 44.2 | mobile NPU, Edge TPU |
| Small | 8.8M | 19.6G | 390 | 49.7 | high-end mobile, embedded GPU |
| Base | 22.4M | 56.0G | 255 | 56.1 | real-time COCO target |
| Large | 41.8M | 108G | 172 | 57.2 | cloud real-time |
| XLarge | 73.5M | 196G | 112 | 58.4 | maximum accuracy |

These are research targets. The repository does not include trained weights or measured COCO logs yet.
