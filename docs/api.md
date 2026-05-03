# API Documentation

Core packages:

- `plas.cli`: unified command-line interface.
- `plas.core.config`: YAML, JSON, TOML loading and CLI overrides.
- `plas.zoo`: model-card registry, checkpoint download, SHA256 verification.
- `plas.engine.inference`: native inference, dynamic batching, async API, video streams.
- `plas.engine.exporters`: ONNX, TensorRT, OpenVINO, CoreML, TFLite, TorchScript, NCNN, Edge TPU dispatch.
- `plas.monitoring`: W&B, TensorBoard, CSV, and JSONL logging.
- `plas.compression`: pruning, sparsity, QAT, PTQ.
- `plas.distill`: logit, feature, relational, and self-distillation losses.
- `plas.tracking`: ByteTrack-style tracking, trajectories, zone counting.
- `plas.explain`: Grad-CAM, Eigen-CAM, attention rollout, failure analysis.
- `plas.data`: dataset validation, conversion, duplicate detection, data-centric utilities.

Install developer docs tooling with:

```bash
pip install -e ".[dev]"
```

Recommended generated docs command:

```bash
pydoc-markdown -I . -m plas -m plasformers --render-toc > docs/generated_api.md
```
