# Migration Guides

## From Ultralytics YOLO

- Replace `yolo train` with `plas train`.
- Replace `model.export(format="onnx")` with `plas export onnx`.
- Use `plas.core.plugins` to register custom losses and augmentations.

## From MMDetection

- Convert dataset configs to YAML/JSON/TOML.
- Register custom modules as Plasformers plugins instead of registry configs.
- Use `plas val` with an evaluator plugin for dataset-specific metrics.

## From Detectron2

- Replace config node mutation with dot-key CLI overrides.
- Use `plas.engine.InferenceEngine` for deployment-facing inference.
- Use `plas.serve.create_app` for FastAPI serving.
