"""Benchmark arena for comparing Plasformers against external systems."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class Competitor:
    """Benchmark competitor entry."""

    name: str
    command: list[str]
    metrics_file: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ArenaResult:
    """Benchmark result record."""

    name: str
    status: str
    elapsed_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


DEFAULT_COMPETITORS = [
    Competitor("YOLOv10", ["python", "-m", "benchmarks.yolov10"]),
    Competitor("RT-DETR", ["python", "-m", "benchmarks.rtdetr"]),
    Competitor("GroundingDINO", ["python", "-m", "benchmarks.grounding_dino"]),
    Competitor("SAM2", ["python", "-m", "benchmarks.sam2"]),
    Competitor("Florence2", ["python", "-m", "benchmarks.florence2"]),
]


class BenchmarkArena:
    """Run benchmark commands and normalize metrics."""

    def __init__(self, competitors: Iterable[Competitor] | None = None) -> None:
        self.competitors = list(competitors or DEFAULT_COMPETITORS)

    def run(self, output: str | Path = "benchmarks/arena_results.json") -> list[ArenaResult]:
        results = [self._run_one(competitor) for competitor in self.competitors]
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
        return results

    def _run_one(self, competitor: Competitor) -> ArenaResult:
        start = time.perf_counter()
        try:
            completed = subprocess.run(competitor.command, capture_output=True, text=True, check=False)
            metrics = self._read_metrics(competitor.metrics_file)
            status = "ok" if completed.returncode == 0 else "failed"
            return ArenaResult(competitor.name, status, time.perf_counter() - start, metrics, completed.stderr or None)
        except OSError as exc:
            return ArenaResult(competitor.name, "unavailable", time.perf_counter() - start, error=str(exc))

    @staticmethod
    def _read_metrics(path: str | None) -> dict[str, float]:
        if not path or not Path(path).exists():
            return {}
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {key: float(value) for key, value in data.items() if isinstance(value, (int, float))}


def latex_benchmark_table(rows: Iterable[Mapping[str, object]]) -> str:
    lines = ["\\begin{tabular}{lrrrr}", "Model & AP & FPS & Params & FLOPs \\\\", "\\hline"]
    for row in rows:
        lines.append(
            f"{row.get('model', '')} & {row.get('ap', '')} & {row.get('fps', '')} & {row.get('params', '')} & {row.get('flops', '')} \\\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines)
