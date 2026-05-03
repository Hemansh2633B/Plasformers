# Plasformers Framework

The enterprise framework layer lives in `plas/` and wraps the research model in
production services.

## CLI

```bash
plas train --config configs/train_coco.yaml --set optimization.base_lr=0.001
plas val --config configs/train_coco.yaml --checkpoint checkpoints/best.pt
plas predict image.jpg --variant nano --output pred.json
plas export onnx --variant base --output exports/base
plas benchmark --all --device cuda
plas profile --variant base
plas tune --method optuna --trials 50
plas prune --variant small --amount 0.2
plas quantize --variant nano --qat
plas distill --config configs/train_coco.yaml --teacher YOLOv10
plas serve --variant nano --port 8000
plas track video.mp4 --tracker bytetrack
plas explain image.jpg --target-layer backbone.stage_modules.4
plas doctor
plas tasks
plas hardware
plas registry list
plas marketplace
plas arena
plas copilot --metrics current.json --baseline previous.json
```

Typer provides completion:

```bash
plas --install-completion
```

## Configuration

YAML, JSON, and TOML files are supported. CLI overrides use dot keys:

```bash
plas train -c configs/train_coco.yaml --set model.variant=small --set optimization.epochs=300
```

## Extension Points

Register plugins for custom backbones, losses, augmentations, heads, exporters,
trainers, evaluators, distillation runners, and tuning objectives.

## Serving

REST and WebSocket serving use FastAPI:

- `GET /healthz`
- `GET /metrics`
- `POST /predict`
- `WS /ws/video`

Bearer authentication is enabled with `plas serve --token <secret>`.

## Model Zoo

`plas zoo` lists versioned model cards. Public weights are intentionally not
claimed until trained checkpoints and SHA256 hashes are published. Internal
teams can register private cards with URLs and hashes.
