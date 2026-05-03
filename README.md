# Plasformers

**Plasformers** is a research-grade object detector design and implementation. The name is a compact acronym for **Progressive Local-Attentive Spectral Fusion with Object-query Routing for Multi-scale Efficient Real-time Sensing**. It captures the central idea: local convolution, compressed global attention, and spectral filtering flow through every stage and are fused by input-adaptive routing.

This repository contains a deployable PyTorch reference implementation, export helpers, training configuration, an architecture specification, an ablation plan, and a paper-style draft.

## Status

The code is implementation-ready, but the headline benchmark numbers in the docs are **targets**, not measured results. They should be treated as hypotheses until reproduced with full COCO training and TensorRT benchmarking.

## Architecture Diagram

```text
Input Bx3xHxW
  |
  v
Adaptive Hybrid Stem
  |- learnable frequency-aware conditioning
  |- anti-aliased stride-2 conv
  |- detail-preserving pooled bypass
  v
S1  stride 4   high-resolution shallow tri-path features
S2  stride 8   low-level semantic tri-path features
S3  stride 16  mid-level abstraction tri-path features
S4  stride 32  deep semantic tri-path features
S5  stride 64  context aggregation tri-path features
  |
  v
Cross-Scale Global Fusion Neck
  |- P3/P4/P5/P6/P7 projections
  |- bidirectional weighted routing
  |- deformable cross-scale resampling
  |- sparse level-token exchange
  v
Dynamic Query-Guided Detection Head
  |- scene-conditioned dynamic depthwise kernels
  |- decoupled classification and regression towers
  |- IoU-aware refinement and DFL box distributions
  |- auxiliary one-to-many branch during training
  v
Unified outputs: boxes, objectness, classes, optional masks, optional keypoints
```

## Quick Start

```bash
pip install -e ".[dev,export,serve,ui]"
plas doctor
plas zoo
python scripts/benchmark.py --variant nano --height 640 --width 640 --amp --export-mode
python scripts/export_onnx.py --variant base --output exports/plasformers_base.onnx
```

```python
from plasformers import build_plasformers

model = build_plasformers("base", num_classes=80)
model.eval()
```

## Training And Deployment

- Unified CLI: `plas train|val|predict|export|benchmark|profile|tune|prune|quantize|distill|serve|track|explain|doctor`
- Foundation platform APIs: multi-task heads, VLM dual encoders, SSL, synthetic data, NAS, FL, continual learning, world models, diffusion, governance, registry, and marketplace
- Training recipe: `configs/train_coco.yaml`
- Domain adaptation profiles: `configs/train_aerial_medical_industrial.yaml`
- AMP, EMA, DDP, and optional XLA hooks: `plasformers/training.py`
- ONNX export: `scripts/export_onnx.py`
- Latency smoke benchmark: `scripts/benchmark.py`
- REST/WebSocket serving: `plas serve`
- Interactive Gradio UI: `plas web`
- Auto-completion: `plas --install-completion`

## Repository Structure

```text
plas/
  cli.py             unified command-line interface
  apps/              Gradio web UI
  compression/       pruning, sparsity, and quantization
  core/              config, security, plugins, doctor, experiments
  data/              dataset validation, conversion, duplicate analysis
  engine/            inference, export, benchmark, profile
  monitoring/        W&B, TensorBoard, CSV, JSONL loggers
  serve/             FastAPI, WebSocket, gRPC hooks
  tracking.py        ByteTrack-style tracker and zone counting
  explain.py         Grad-CAM, Eigen-CAM, attention rollout
  foundation/        universal tasks and multimodal models
  agents/            agentic visual workflows
  research/          SSL, synthetic data, NAS, FL, continual, RL, world, diffusion
  runtime/           hardware, distributed, TPU, compiler, autoscaling
  enterprise/        security, governance, registry, collaboration
  ecosystem/         marketplace, benchmark arena, copilot, academic, DX
plasformers/
  config.py          scaling family definitions
  modules.py         stem, tri-path blocks, spectral path, neck primitives
  model.py           end-to-end detector
  losses.py          Varifocal, DFL, CIoU/SIoU, consistency utilities
  export.py          ONNX and TorchScript export wrappers
  training.py        AMP, EMA, DDP, and XLA helpers
  optimization.py    fusion, QAT, pruning, and sparsity helpers
configs/
  plasformers_family.yaml
  train_coco.yaml
  train_aerial_medical_industrial.yaml
docs/
  architecture.md
  framework.md
  paper.md
  ablation_plan.md
deployments/
  docker/
  kubernetes/
  mobile/
examples/
benchmarks/
plugins/
notebooks/
scripts/
  export_onnx.py
  benchmark.py
tests/
  test_smoke.py
```

## Baseline Context

The design target is motivated by the speed/accuracy frontier reported by recent detectors:

- YOLOv10 reports NMS-free real-time detection and strong efficiency across scales: https://arxiv.org/abs/2405.14458
- RT-DETR reports 53.1 AP at 108 FPS on T4 for R50 and 54.3 AP at 74 FPS for R101: https://arxiv.org/abs/2304.08069
- YOLOX reports anchor-free decoupled-head improvements and broad deployment support: https://arxiv.org/abs/2107.08430

Plasformers differs architecturally by making spectral processing a first-class branch at every stage, using per-stage tri-path adaptive fusion, and replacing hand-designed pyramid addition with learned sparse cross-scale routing.

## Suggested Future Improvements

- Add a full COCO dataloader and task-aligned assigner implementation.
- Train proxy-calibrated FFT export mode with a distillation loss to reduce eager/export accuracy drift.
- Add TensorRT plugin-free decode and batched NMS-free postprocessing.
- Add QAT recipes for Edge TPU, OpenVINO INT8, and CoreML palettization.
- Publish measured COCO, VisDrone, DOTA, MVTec, and medical adaptation benchmarks with logs and checkpoints.
