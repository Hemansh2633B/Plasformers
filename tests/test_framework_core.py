"""Framework core tests that do not require PyTorch."""

from __future__ import annotations

from pathlib import Path

from plas.core.config import load_with_overrides, parse_overrides
from plas.core.doctor import run_doctor
from plas.core.plugins import PluginRegistry
from plas.zoo import get_model_zoo


def test_config_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text('{"model": {"variant": "nano"}, "optimization": {"epochs": 10}}', encoding="utf-8")
    bundle = load_with_overrides(cfg, ["model.variant=small", "optimization.base_lr=0.001"])
    assert bundle.data["model"]["variant"] == "small"
    assert bundle.data["optimization"]["epochs"] == 10
    assert bundle.data["optimization"]["base_lr"] == 0.001


def test_parse_overrides_json_scalars() -> None:
    parsed = parse_overrides(["a.b=true", "a.c=[1,2]", "x=null"])
    assert parsed["a"]["b"] is True
    assert parsed["a"]["c"] == [1, 2]
    assert parsed["x"] is None


def test_model_zoo_cards() -> None:
    zoo = get_model_zoo()
    names = {card.name for card in zoo.list()}
    assert "plasformers-nano" in names
    assert zoo.get("plasformers-base").variant == "base"


def test_plugin_registry() -> None:
    registry = PluginRegistry()

    def factory() -> str:
        return "ok"

    registry.register("loss", "toy", factory)
    assert registry.get("loss", "toy")() == "ok"


def test_doctor_report_shape() -> None:
    report = run_doctor()
    assert "python" in report
    assert "packages" in report
    assert "tools" in report
