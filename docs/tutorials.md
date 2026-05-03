# Tutorials

## Train

```bash
plas train -c configs/train_coco.yaml --set model.variant=small
```

For full dataset-specific training, register a `train_runner` plugin and set:

```yaml
runner: my_runner
```

## Validate

```bash
plas val -c configs/train_coco.yaml --checkpoint checkpoints/best.pt
```

## Export

```bash
plas export onnx --variant base --output exports/base
plas export tensorrt --variant base --output exports/base --int8
```

## Serve

```bash
plas serve --variant nano --port 8000 --token secret
```

## Web UI

```bash
plas web --variant nano
```
