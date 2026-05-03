# Coral TPU Deployment

Recommended path:

1. Train or fine-tune `plasformers-nano` with QAT.
2. `plas export edgetpu --variant nano --int8`
3. Validate the compiled Edge TPU artifact against a calibration split.
