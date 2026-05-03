"""gRPC serving scaffold.

Production deployments can generate typed stubs from their own proto files and
bind them to `InferenceEngine`. This module keeps the framework dependency
optional while providing a runnable health service hook.
"""

from __future__ import annotations

from concurrent import futures
from typing import Any


def create_grpc_server(max_workers: int = 8) -> Any:
    """Create a bare gRPC server for custom service registration."""

    try:
        import grpc
    except ImportError as exc:
        raise RuntimeError("gRPC serving requires grpcio") from exc
    return grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
