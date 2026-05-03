"""Gradio web interface for Plasformers."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from plas.engine.inference import InferenceEngine


def _draw_detections(image: np.ndarray, detections: list[Any]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return image
    pil = Image.fromarray(image.astype("uint8")).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x1, y1, x2, y2 = det.box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{det.class_id}:{det.score:.2f}", fill="red")
    return np.asarray(pil)


def build_demo(default_variant: str = "nano") -> Any:
    """Build a Gradio Blocks app."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Web UI requires gradio. Install plasformers[ui].") from exc

    engines: dict[str, InferenceEngine] = {}

    def get_engine(variant: str, threshold: float) -> InferenceEngine:
        key = f"{variant}:{threshold:.3f}"
        if key not in engines:
            engines[key] = InferenceEngine(variant=variant, conf_threshold=threshold)
        return engines[key]

    def infer(image: np.ndarray, variant: str, threshold: float) -> tuple[np.ndarray, str]:
        if image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8), "No image provided."
        start = time.perf_counter()
        result = get_engine(variant, threshold).predict(image)
        rendered = _draw_detections(image, result.detections)
        summary = {
            "fps": result.fps,
            "latency_ms": result.latency_ms,
            "wall_ms": (time.perf_counter() - start) * 1000.0,
            "detections": [asdict(det) for det in result.detections],
        }
        return rendered, str(summary)

    def compare(image: np.ndarray, left: str, right: str, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        l_result = get_engine(left, threshold).predict(image)
        r_result = get_engine(right, threshold).predict(image)
        return _draw_detections(image, l_result.detections), _draw_detections(image, r_result.detections)

    def video_detect(video_path: str, variant: str, threshold: float) -> str:
        if not video_path:
            return "No video provided."
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video detection requires opencv-python") from exc
        engine = get_engine(variant, threshold)
        cap = cv2.VideoCapture(video_path)
        frames = 0
        detections = 0
        start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = engine.predict(rgb)
            frames += 1
            detections += len(result.detections)
        cap.release()
        elapsed = time.perf_counter() - start
        return str({"file": str(Path(video_path).name), "frames": frames, "detections": detections, "fps": frames / max(elapsed, 1e-6)})

    def explain_image(image: np.ndarray, variant: str, target_layer: str) -> np.ndarray:
        if image is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        import torch
        from plas.engine.inference import _preprocess
        from plas.explain import GradCAM
        from plasformers import build_plasformers

        model = build_plasformers(variant).eval()
        tensor, _ = _preprocess([image], (640, 640), "cpu", False)
        explanation = GradCAM(model, target_layer)(tensor)
        heatmap = explanation.heatmap[0, 0].detach().cpu().numpy()
        heatmap = (heatmap * 255).clip(0, 255).astype("uint8")
        rgb = np.stack([heatmap, np.zeros_like(heatmap), 255 - heatmap], axis=-1)
        return rgb

    with gr.Blocks(title="Plasformers Studio") as demo:
        gr.Markdown("# Plasformers Studio")
        with gr.Tab("Image"):
            image = gr.Image(type="numpy", label="Drag image here")
            variant = gr.Dropdown(["nano", "small", "base", "large", "xlarge"], value=default_variant, label="Model")
            threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
            run = gr.Button("Run Detection")
            output = gr.Image(type="numpy", label="Detections")
            summary = gr.Textbox(label="FPS and detections")
            run.click(infer, inputs=[image, variant, threshold], outputs=[output, summary])
        with gr.Tab("Compare"):
            cmp_image = gr.Image(type="numpy", label="Image")
            left = gr.Dropdown(["nano", "small", "base", "large", "xlarge"], value="nano", label="Left")
            right = gr.Dropdown(["nano", "small", "base", "large", "xlarge"], value="small", label="Right")
            cmp_threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
            cmp_run = gr.Button("Compare")
            left_out = gr.Image(type="numpy", label="Left detections")
            right_out = gr.Image(type="numpy", label="Right detections")
            cmp_run.click(compare, inputs=[cmp_image, left, right, cmp_threshold], outputs=[left_out, right_out])
        with gr.Tab("Video And Webcam"):
            video = gr.Video(label="Video")
            video_variant = gr.Dropdown(["nano", "small", "base", "large", "xlarge"], value=default_variant, label="Model")
            video_threshold = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
            video_run = gr.Button("Run Video Detection")
            video_summary = gr.Textbox(label="Video FPS and detections")
            video_run.click(video_detect, inputs=[video, video_variant, video_threshold], outputs=[video_summary])
            webcam = gr.Image(type="numpy", sources=["webcam"], label="Webcam frame")
            webcam_out = gr.Image(type="numpy", label="Webcam detections")
            webcam_text = gr.Textbox(label="Webcam FPS")
            webcam.change(infer, inputs=[webcam, video_variant, video_threshold], outputs=[webcam_out, webcam_text])
        with gr.Tab("Attention Maps"):
            attn_image = gr.Image(type="numpy", label="Image")
            attn_variant = gr.Dropdown(["nano", "small", "base", "large", "xlarge"], value=default_variant, label="Model")
            attn_layer = gr.Textbox(value="backbone.stage_modules.4", label="Target layer")
            attn_run = gr.Button("Generate Grad-CAM")
            attn_out = gr.Image(type="numpy", label="Attention map")
            attn_run.click(explain_image, inputs=[attn_image, attn_variant, attn_layer], outputs=[attn_out])
    return demo


def launch(default_variant: str = "nano", host: str = "127.0.0.1", port: int = 7860) -> None:
    demo = build_demo(default_variant)
    demo.launch(server_name=host, server_port=port)
