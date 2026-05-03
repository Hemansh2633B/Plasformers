"""Ray Serve deployment for Plasformers."""

from __future__ import annotations

from ray import serve

from plas.engine.inference import InferenceEngine


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 2})
class PlasformersDeployment:
    def __init__(self, variant: str = "nano") -> None:
        self.engine = InferenceEngine(variant=variant)

    async def __call__(self, request):
        payload = await request.json()
        result = self.engine.predict(payload["image"])
        return {"detections": [det.__dict__ for det in result.detections], "latency_ms": result.latency_ms}


app = PlasformersDeployment.bind()
