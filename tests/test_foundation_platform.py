"""Foundation-platform tests that do not require PyTorch."""

from __future__ import annotations

from pathlib import Path

from plas.agents.runtime import VisionAgent, VisionTool
from plas.ecosystem.assistant import ResearchCopilot
from plas.ecosystem.marketplace import MarketplaceIndex, MarketplaceItem
from plas.enterprise.governance import DatasetLineageRecord, dataset_fingerprint, pii_scan_text
from plas.enterprise.security import sign_artifact, verify_artifact_signature
from plas.foundation.tasks import VisionTask, get_default_task_specs
from plas.research.nas import ArchitectureCandidate, HardwareAwareNAS, ParetoPoint
from plas.research.synthetic import ObjectAsset, SyntheticDataFactory
from plas.runtime.hardware import optimal_runtime


def test_default_task_specs_cover_foundation_tasks() -> None:
    specs = get_default_task_specs()
    assert VisionTask.DETECTION in specs
    assert VisionTask.VQA in specs
    assert VisionTask.VIDEO_UNDERSTANDING in specs


def test_vision_agent_runs_plan() -> None:
    agent = VisionAgent([VisionTool("step", "test", lambda args: {"value": args.get("seed", 0) + 1})])
    result = agent.run_plan("test", [{"tool": "step"}], {"seed": 2})
    assert result.artifacts["value"] == 3


def test_synthetic_factory_manifest(tmp_path: Path) -> None:
    factory = SyntheticDataFactory(tmp_path)
    scene = factory.sample_scene([ObjectAsset("box", class_id=1)])
    manifest = factory.write_manifest([scene])
    assert manifest.exists()


def test_nas_pareto() -> None:
    def evaluator(candidate: ArchitectureCandidate) -> ParetoPoint:
        return ParetoPoint(candidate, accuracy=candidate.width_mult, latency_ms=10 / candidate.width_mult, memory_mb=1, device="cpu")

    frontier = HardwareAwareNAS(evaluator, latency_budget_ms=100).search(trials=4)
    assert frontier


def test_enterprise_security_and_governance(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_text("weights", encoding="utf-8")
    signed = sign_artifact(artifact, "secret")
    assert verify_artifact_signature(signed, "secret")
    assert pii_scan_text("a@example.com")["email"]
    lineage = DatasetLineageRecord("dataset", "1.0.0", "unit", dataset_fingerprint([artifact]))
    assert lineage.fingerprint


def test_marketplace_and_copilot(tmp_path: Path) -> None:
    index = MarketplaceIndex(tmp_path / "index.json")
    index.add(MarketplaceItem("toy", "plugin", "0.1.0", "Toy plugin", "local"))
    assert index.search("toy")
    summary = ResearchCopilot().summarize({"ap": 1.0}, {"ap": 1.1})
    assert summary.regressions


def test_runtime_fallback() -> None:
    assert isinstance(optimal_runtime(), str)
