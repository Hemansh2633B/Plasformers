"""MLOps integrations for MLflow, Ray, DVC, Hugging Face Hub, Docker, and Kubernetes."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping


def log_mlflow_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow integration requires mlflow") from exc
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def run_ray_train(trainable: object, config: Mapping[str, object]) -> object:
    try:
        from ray import tune
    except ImportError as exc:
        raise RuntimeError("Ray integration requires ray[tune]") from exc
    return tune.Tuner(trainable, param_space=dict(config)).fit()


def dvc_track(path: str | Path) -> None:
    subprocess.run(["dvc", "add", str(path)], check=True)


def push_to_hub(repo_id: str, folder: str | Path, commit_message: str = "Upload Plasformers artifact") -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Hugging Face Hub integration requires huggingface_hub") from exc
    api = HfApi()
    return api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message=commit_message)


def docker_build(tag: str = "plasformers:latest", dockerfile: str = "deployments/docker/Dockerfile") -> None:
    subprocess.run(["docker", "build", "-t", tag, "-f", dockerfile, "."], check=True)


def kubectl_apply(manifest: str | Path) -> None:
    subprocess.run(["kubectl", "apply", "-f", str(manifest)], check=True)
