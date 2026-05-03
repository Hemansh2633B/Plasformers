"""FastAPI serving application."""

from __future__ import annotations

import io
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from plas.engine.inference import InferenceEngine


def create_app(
    *,
    variant: str = "nano",
    checkpoint: str | None = None,
    device: str = "auto",
    auth_token: str | None = None,
) -> Any:
    """Create a FastAPI app with health, metrics, batch inference, and WebSocket endpoints."""

    try:
        from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket
    except ImportError as exc:
        raise RuntimeError("Serving requires fastapi. Install plasformers[serve].") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Serving image uploads requires Pillow") from exc

    app = FastAPI(title="Plasformers Serving", version="0.1.0")
    engine: InferenceEngine | None = None
    metrics = {"requests": 0, "images": 0, "total_latency_ms": 0.0, "started_at": time.time()}

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if auth_token is None:
            return
        if authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    def get_engine() -> InferenceEngine:
        nonlocal engine
        if engine is None:
            engine = InferenceEngine(variant=variant, checkpoint=checkpoint, device=device)
        return engine

    async def read_image(upload: UploadFile) -> np.ndarray:
        raw = await upload.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(image)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "variant": variant, "uptime_s": time.time() - metrics["started_at"]}

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        avg = metrics["total_latency_ms"] / max(1, metrics["images"])
        return {**metrics, "avg_latency_ms": avg}

    @app.post("/predict", dependencies=[Depends(require_auth)])
    async def predict(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        images = [await read_image(file) for file in files]
        results = get_engine().predict_batch(images)
        metrics["requests"] += 1
        metrics["images"] += len(results)
        metrics["total_latency_ms"] += sum(result.latency_ms for result in results)
        return {
            "results": [
                {
                    "latency_ms": result.latency_ms,
                    "fps": result.fps,
                    "detections": [asdict(det) for det in result.detections],
                }
                for result in results
            ]
        }

    @app.websocket("/ws/video")
    async def video_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            data = await websocket.receive_bytes()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            result = get_engine().predict(np.asarray(image))
            await websocket.send_json(
                {
                    "latency_ms": result.latency_ms,
                    "fps": result.fps,
                    "detections": [asdict(det) for det in result.detections],
                }
            )

    return app
