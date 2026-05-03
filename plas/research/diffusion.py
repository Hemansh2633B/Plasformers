"""Diffusion model support for generation and synthetic data refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DiffusionRequest:
    """Generation request."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    mode: str = "text_to_image"
    mask: Any | None = None
    detections: Any | None = None


class DiffusionPipelineAdapter:
    """Adapter around diffusers-style pipelines."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def generate(self, request: DiffusionRequest) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
        }
        if request.mask is not None:
            kwargs["mask_image"] = request.mask
        return self.pipeline(**kwargs)


def detection_guidance_prompt(base_prompt: str, detections: list[Mapping[str, Any]]) -> str:
    objects = ", ".join(str(det.get("label", det.get("class_id", "object"))) for det in detections)
    return f"{base_prompt}. Preserve and refine detected objects: {objects}."


def super_resolution_request(image: Any, scale: int = 4) -> dict[str, Any]:
    return {"mode": "super_resolution", "image": image, "scale": scale}
