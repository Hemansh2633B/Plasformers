# Plasformers Vision OS

Plasformers now includes a foundation-platform layer intended to grow beyond
object detection into a unified computer vision operating system.

## Universal Tasks

`plas.foundation.tasks` defines a shared task taxonomy and task-head builders for:

- classification
- detection
- instance, semantic, and panoptic segmentation
- pose estimation
- oriented boxes
- tracking
- visual grounding and referring expressions
- captioning, OCR, document understanding, VQA
- depth, optical flow, 3D detection, video understanding

## Multimodal

`plas.foundation.multimodal` provides CLIP-style dual encoders, contrastive loss,
image-text retrieval primitives, grounding heads, promptable detection queries,
and open-vocabulary scoring hooks.

## Research Systems

`plas.research` includes:

- masked autoencoding, contrastive learning, DINO, BYOL, SimCLR, MoCo, I-JEPA losses
- synthetic data factory with Blender, Unreal, physics, and auto-label hooks
- hardware-aware NAS with Pareto frontier optimization
- federated learning with secure aggregation and DP hooks
- continual learning with replay buffers and EWC
- RL controllers for augmentation and architecture search
- latent world-model components
- diffusion adapters for generation, editing, inpainting, super-resolution, and detection-guided refinement

## Runtime

`plas.runtime` includes hardware detection, distributed training plans, FSDP,
DeepSpeed, TPU PJRT/XLA helpers, TorchInductor, CUDA graphs, FlashAttention
checks, fused optimizer hooks, dynamic batching, HPA manifests, and serverless
cold-start plans.

## Enterprise

`plas.enterprise` includes signed artifacts, model watermarking payloads, secure
enclave requests, dataset lineage, audit trails, privacy scanning, compliance
reports, a filesystem-backed model registry, rollback, promotion, and team
workspace primitives.

## Ecosystem

`plas.ecosystem` includes a marketplace index, benchmark arena against external
systems, research copilot, academic appendix/citation/table utilities, cloud
marketplace listings, and developer-experience helpers for graph exploration and
live profiling dashboards.

## Discovery Commands

```bash
plas tasks
plas hardware
plas marketplace
plas registry list
plas arena
plas copilot --metrics current.json --baseline previous.json
```
