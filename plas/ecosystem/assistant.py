"""Auto research assistant for experiment planning and regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass
class ExperimentSummary:
    """Generated experiment summary."""

    title: str
    findings: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    next_experiments: list[str] = field(default_factory=list)


class ResearchCopilot:
    """Rule-based research copilot with LLM-ready structured outputs."""

    def suggest_ablations(self, components: Iterable[str]) -> list[dict[str, object]]:
        return [
            {"name": f"remove_{component}", "change": {component: "disabled"}, "expected_signal": "delta_ap_latency"}
            for component in components
        ]

    def suggest_hyperparameters(self, baseline: Mapping[str, object]) -> list[dict[str, object]]:
        lr = float(baseline.get("base_lr", 1e-3))
        return [
            {"optimization.base_lr": lr * 0.5},
            {"optimization.base_lr": lr * 1.5},
            {"loss.dfl_weight": 2.0},
            {"augmentation.copy_paste.prob": 0.45},
        ]

    def detect_regressions(self, current: Mapping[str, float], baseline: Mapping[str, float]) -> list[str]:
        regressions = []
        for metric, old_value in baseline.items():
            if metric in current:
                new_value = current[metric]
                if metric.lower().endswith(("ap", "fps")) and new_value < old_value * 0.99:
                    regressions.append(f"{metric} dropped from {old_value:.4f} to {new_value:.4f}")
                if "latency" in metric.lower() and new_value > old_value * 1.01:
                    regressions.append(f"{metric} increased from {old_value:.4f} to {new_value:.4f}")
        return regressions

    def summarize(self, metrics: Mapping[str, float], baseline: Mapping[str, float] | None = None) -> ExperimentSummary:
        regressions = self.detect_regressions(metrics, baseline or {})
        findings = [f"{key}: {value:.4f}" for key, value in metrics.items()]
        next_steps = ["Run component ablations", "Validate export-mode parity", "Benchmark on target hardware"]
        return ExperimentSummary("Plasformers Experiment Summary", findings, regressions, next_steps)

    def write_summary(self, summary: ExperimentSummary, output: str | Path) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
        return out
