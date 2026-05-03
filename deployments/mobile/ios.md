# iOS Deployment

Recommended path:

1. `plas export coreml --variant nano`
2. Use static input sizes for best Neural Engine compilation.
3. Apply CoreML palettization or INT8 linear quantization after validation.
