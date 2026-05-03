# Jetson Deployment

Recommended path:

1. `plas export tensorrt --variant small --int8`
2. Build with JetPack-compatible TensorRT.
3. Serve with `plas serve` or embed `plas.engine.InferenceEngine`.
