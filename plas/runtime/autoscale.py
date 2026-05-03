"""Auto-scaling inference and dynamic batching."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List


@dataclass(frozen=True)
class AutoscalePolicy:
    """Horizontal autoscaling policy."""

    min_replicas: int = 1
    max_replicas: int = 10
    target_qps: float = 20.0
    target_latency_ms: float = 50.0
    scale_down_cooldown_s: int = 120


class DynamicBatcher:
    """Async dynamic batcher for serverless or always-on inference."""

    def __init__(self, infer: Callable[[list[Any]], Awaitable[list[Any]]], max_batch: int = 16, timeout_ms: float = 5.0) -> None:
        self.infer = infer
        self.max_batch = max_batch
        self.timeout_ms = timeout_ms
        self.queue: list[tuple[Any, asyncio.Future[Any]]] = []
        self.lock = asyncio.Lock()

    async def submit(self, item: Any) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        async with self.lock:
            self.queue.append((item, future))
            if len(self.queue) >= self.max_batch:
                await self._flush_locked()
            else:
                loop.call_later(self.timeout_ms / 1000.0, lambda: asyncio.create_task(self.flush()))
        return await future

    async def flush(self) -> None:
        async with self.lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self.queue:
            return
        batch, futures = zip(*self.queue)
        self.queue = []
        started = time.perf_counter()
        outputs = await self.infer(list(batch))
        for future, output in zip(futures, outputs):
            if not future.done():
                future.set_result({"output": output, "queue_latency_ms": (time.perf_counter() - started) * 1000.0})


def kubernetes_hpa_manifest(name: str, policy: AutoscalePolicy) -> dict[str, Any]:
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": name},
        "spec": {
            "minReplicas": policy.min_replicas,
            "maxReplicas": policy.max_replicas,
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 70}},
                }
            ],
        },
    }


def serverless_cold_start_plan() -> dict[str, Any]:
    return {"preload_model": True, "warm_min_instances": 1, "snapshot_weights": True, "lazy_cuda_graph": True}
