"""Dashboard integrations for W&B, TensorBoard, CSV, and JSONL."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Protocol


class MetricLogger(Protocol):
    """Logger protocol used by training loops."""

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        ...

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        ...

    def close(self) -> None:
        ...


class CsvLogger:
    """Append scalar metrics to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({key: row.get(key, "") for key in self._fieldnames})

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        return

    def close(self) -> None:
        return


class JsonlLogger:
    """Append metrics and artifacts as JSON lines."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "metrics", "step": step, "metrics": dict(metrics)}) + "\n")

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "artifact", "name": name, "path": str(path)}) + "\n")

    def close(self) -> None:
        return


class TensorBoardLogger:
    """TensorBoard scalar, image, histogram, and PR-curve logger."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard logging requires torch and tensorboard") from exc
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        self.writer.add_text(name or "artifact", str(path))

    def log_image(self, tag: str, image: Any, step: int) -> None:
        self.writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self.writer.add_histogram(tag, values, step)

    def close(self) -> None:
        self.writer.close()


class WandbLogger:
    """Weights & Biases logger."""

    def __init__(self, project: str, name: str | None = None, config: Mapping[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Weights & Biases logging requires the optional 'wandb' package") from exc
        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=dict(config or {}))

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.wandb.log(dict(metrics), step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        artifact = self.wandb.Artifact(name or Path(path).stem, type="artifact")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def close(self) -> None:
        self.run.finish()


class CompositeLogger:
    """Fan-out logger."""

    def __init__(self, *loggers: MetricLogger) -> None:
        self.loggers = loggers

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        for logger in self.loggers:
            logger.log_artifact(path, name)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
