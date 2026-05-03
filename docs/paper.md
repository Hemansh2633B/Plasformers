# Plasformers: Progressive Local-Attentive Spectral Fusion for Real-Time Object Detection

## Abstract

Real-time object detection is constrained by an accuracy-latency trade-off that becomes especially severe for small objects, degraded imagery, and edge deployment. We present **Plasformers**, a detector built from first principles around three complementary information streams: local convolutional routing, compressed global attention, and frequency-domain spectral gating. Each backbone block fuses the streams with input-adaptive simplex weights, while a cross-scale global fusion neck performs bidirectional pyramid routing, deformable resampling, and sparse token exchange across P3-P7. A dynamic query-guided head conditions decoupled prediction kernels on scene context and supports IoU-aware localization, Distribution Focal Loss, and auxiliary one-to-many training. Plasformers is designed for efficient mixed-precision training and practical deployment through ONNX, TensorRT, CoreML, OpenVINO, and TPU-compatible export paths. We target a new real-time frontier above 56 COCO AP at more than 250 TensorRT FP16 FPS on T4 for the base configuration, pending full empirical validation.

## 1. Introduction

Modern real-time detectors are highly optimized, yet their dominant design patterns often emphasize either convolutional locality or transformer context. Small objects, low light, occlusion, blur, and scale variation require a broader signal model: fine spatial evidence, long-range semantics, and frequency-aware denoising must cooperate throughout the network.

Plasformers introduces a tri-path hierarchy. The local path preserves efficient convolutional priors; the global path provides compressed cross-window attention; the spectral path explicitly manipulates frequency bands using learnable masks. Rather than stacking these mechanisms in separate regions of the network, Plasformers fuses all three at every block with sample-adaptive gates.

The architecture is designed for COCO, aerial imagery, medical imaging, industrial defects, and autonomous systems. Export compatibility is a first-order constraint, so training-only operators have deterministic deployment fallbacks.

## 2. Methodology

### 2.1 Adaptive Hybrid Stem

Given input `X in R^{B x 3 x H x W}`, the stem first applies a learnable frequency-aware residual conditioner:

`X' = X + sigmoid(g(GAP(X))) * W_1 DW_3(X)`.

Two anti-aliased stride-2 convolutions produce `ceil(H/4) x ceil(W/4)` features. A parallel average-pooled detail bypass is projected and fused with the decimated stream. This reduces aliasing while retaining shallow spatial evidence.

### 2.2 Tri-Path Hierarchical Backbone

Each block computes:

`F_l = L(X)`, `F_g = G(X)`, `F_s = S(X)`,

where `L` is local CSP-style dynamic convolution, `G` is compressed global attention, and `S` is spectral gating.

Fusion is:

`[alpha, beta, gamma] = softmax(W_2 sigma(W_1 GAP(X)))`

`F = alpha F_l + beta F_g + gamma F_s`

`Y = X + eta phi(F)`.

The simplex constraint stabilizes training because branch magnitudes remain bounded.

### 2.3 Local Path

The local path partially routes channels:

`X = [X_a, X_b]`

`L(X) = X + P([X_a, R(X_b)])`.

The routed branch uses dynamic depthwise kernels:

`R(X_b) = BN(SiLU(sum_i pi_i DW_{k_i}(E(X_b))))`.

The kernel weights `pi_i` are conditioned on global pooled context.

### 2.4 Global Path

Full self-attention is avoided. Queries are computed on the full feature map, while keys and values are computed on pooled token grids:

`Q = W_q X`, `K,V = W_{kv} Pool_t(X)`.

`G(X) = X + W_o softmax(QK^T / sqrt(d)) V + DWPE(X)`.

This yields cross-window communication with cost `O(HW t^2 C)` rather than `O((HW)^2 C)`.

### 2.5 Frequency Path

The spectral branch applies:

`Z = FFT2(X)`.

For channel `c`, the mask is:

`M_c(u,v) = sum_b softmax(a_{c,b}) (1 + tanh(g_{c,b})) B_b(u,v)`.

The output is:

`S(X) = X + SE(IFFT2(Z * (1 + M)))`.

During export, the FFT path is replaced by a calibrated depthwise separable proxy.

### 2.6 Cross-Scale Global Fusion Neck

Backbone features become P3-P7. Top-down and bottom-up passes use positive normalized routing:

`P_i = sum_j relu(w_j) F_j / (epsilon + sum_j relu(w_j))`.

Cross-scale token exchange forms descriptors `D_l = GAP(P_l)` and routing:

`A_l = softmax(W_r D_l / tau)`.

`C_l = sum_j A_{l,j} D_j`.

`P_l' = P_l * (1 + sigmoid(W_c C_l))`.

Continuous-offset resampling is used in training for deformable alignment and replaced by bilinear resampling in export mode.

### 2.7 Dynamic Query-Guided Head

The head computes a scene query:

`q = MLP([GAP(P3), ..., GAP(P7)])`.

For each level:

`H_l = P_l + sum_i softmax(W_g q)_i DW_i(P_l)`.

Prediction is decoupled into classification, box distribution, objectness, and IoU refinement. Optional mask coefficients and keypoints share the same scene-conditioned representation.

## 3. Training

We train from scratch using AdamW, EMA, mixed precision, dynamic scaling, Mosaic, Copy-Paste, MixUp, and curriculum image sizing. The final stage disables heavy composition augmentations to recover natural image statistics.

Loss:

`L = lambda_cls L_vfl + lambda_box L_siou + lambda_dfl L_dfl + lambda_obj L_obj + lambda_aux L_consistency`.

Label assignment uses task-aligned matching for the primary branch and one-to-many assignment for the auxiliary branch. The auxiliary branch is removed at inference.

## 4. Experiments

### Datasets

- COCO 2017 for primary detection.
- DOTA and VisDrone for aerial imagery.
- Medical lesion detection datasets adapted to boxes.
- MVTec-style industrial defect localization.
- Autonomous driving datasets with blur, night, weather, and occlusion splits.

### Metrics

We report AP, AP50, AP75, APS, APM, APL, params, FLOPs, memory, TensorRT FP16 FPS on T4, ONNX Runtime FPS, CoreML latency, and TPU v5e throughput.

### Target Benchmark Table

| Model | Params | FLOPs | COCO AP | T4 TRT FP16 FPS | Status |
|---|---:|---:|---:|---:|---|
| Plasformers-N | 3.9M | 8.7G | 44.2 | 520 | target |
| Plasformers-S | 8.8M | 19.6G | 49.7 | 390 | target |
| Plasformers-B | 22.4M | 56.0G | 56.1 | 255 | target |
| Plasformers-L | 41.8M | 108G | 57.2 | 172 | target |
| Plasformers-XL | 73.5M | 196G | 58.4 | 112 | target |

## 5. Ablation Studies

We isolate the frequency branch, compressed attention, adaptive fusion gates, cross-scale neck, dynamic head, IoU refinement, and auxiliary branch. The key expected outcomes are:

- Frequency branch improves APS and corruption robustness.
- Compressed attention improves occlusion and large-object context.
- Adaptive gates improve domain transfer.
- Cross-scale sparse exchange improves small-object recall.
- Dynamic head improves dense scene calibration and AP75.

Full experimental grids are listed in `docs/ablation_plan.md`.

## 6. Deployment

Plasformers uses accelerator-friendly operators by default: convolutions, BatchNorm, SiLU, adaptive pooling, softmax, interpolation, and linear layers. Export mode replaces FFT and continuous offsets. INT8 QAT is recommended for Edge TPU, OpenVINO, and mobile NPU deployment.

## 7. Conclusion

Plasformers is a cohesive detector design that treats local structure, global context, and spectral evidence as equal first-class signals. Its tri-path adaptive backbone, sparse cross-scale neck, and scene-conditioned head are designed to push the real-time detection frontier while preserving practical deployment compatibility. The next step is full-scale empirical validation on COCO and target domains.

## References

- YOLOv10: Real-Time End-to-End Object Detection, https://arxiv.org/abs/2405.14458
- DETRs Beat YOLOs on Real-time Object Detection, https://arxiv.org/abs/2304.08069
- YOLOX: Exceeding YOLO Series in 2021, https://arxiv.org/abs/2107.08430
