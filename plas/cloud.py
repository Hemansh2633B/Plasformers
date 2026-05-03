"""Cloud training launchers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CloudLaunchRequest:
    """Cloud launch request."""

    provider: str
    command: Sequence[str]
    project: str | None = None
    region: str | None = None
    accelerator: str | None = None


def launch_cloud(request: CloudLaunchRequest, dry_run: bool = False) -> list[str]:
    """Build and optionally execute a cloud launch command."""

    provider = request.provider.lower()
    if provider == "aws":
        cmd = ["aws", "sagemaker", "create-training-job"] + list(request.command)
    elif provider == "gcp":
        cmd = ["gcloud", "ai", "custom-jobs", "create"] + list(request.command)
    elif provider == "azure":
        cmd = ["az", "ml", "job", "create"] + list(request.command)
    elif provider == "kaggle":
        cmd = ["kaggle", "kernels", "push"] + list(request.command)
    elif provider == "colab":
        cmd = ["python", "-m", "plas.tools.colab"] + list(request.command)
    elif provider in {"tpu", "tpu-vm"}:
        cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh"] + list(request.command)
    else:
        raise ValueError(f"Unsupported provider: {request.provider}")
    if not dry_run:
        subprocess.run(cmd, check=True)
    return cmd
