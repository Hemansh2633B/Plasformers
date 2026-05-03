"""Unified `plas` command-line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import typer

from .core.config import ConfigBundle, load_with_overrides
from .core.doctor import run_doctor
from .core.experiment import ExperimentRun, set_seed
from .core.plugins import GLOBAL_PLUGINS
from .core.ui import print_panel, print_table, status
from .zoo import get_model_zoo


app = typer.Typer(
    name="plas",
    help="Plasformers enterprise detection framework.",
    add_completion=True,
    no_args_is_help=True,
)


def _bundle(config: Optional[Path], overrides: List[str]) -> ConfigBundle:
    return load_with_overrides(config, overrides)


def _write_json(data: object, output: Optional[Path]) -> None:
    text = json.dumps(data, indent=2, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        typer.echo(text)


def _maybe_torchrun(nproc: int, argv: list[str]) -> None:
    if nproc <= 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"] + argv
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


@app.command()
def train(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML/JSON/TOML config."),
    set_: List[str] = typer.Option([], "--set", help="Override config values, e.g. model.variant=small."),
    launcher: str = typer.Option("none", help="none, torchrun, aws, gcp, azure, kaggle, colab, tpu-vm."),
    nproc: int = typer.Option(1, help="Processes per node for torchrun."),
    dry_run: bool = typer.Option(False, help="Validate and snapshot without training."),
) -> None:
    """Train a detector with snapshots, logging, and plugin runner support."""

    if launcher == "torchrun":
        argv = ["-m", "plas.cli", "train"]
        if config is not None:
            argv.extend(["--config", str(config)])
        for override in set_:
            argv.extend(["--set", override])
        _maybe_torchrun(nproc, argv)
    bundle = _bundle(config, set_)
    seed = int(bundle.data.get("seed", 42))
    set_seed(seed, deterministic=bool(bundle.data.get("deterministic", False)))
    run = ExperimentRun(str(bundle.data.get("experiment", "plas_train")))
    run.snapshot(bundle.data)
    if dry_run:
        print_panel("Train", f"Configuration snapshot written to {run.path}")
        return
    runner_name = str(bundle.data.get("runner", "default"))
    if runner_name != "default":
        runner = GLOBAL_PLUGINS.get("train_runner", runner_name)
        runner(bundle.data, run)
        return
    print_panel(
        "Train",
        "Framework services are ready. Configure `runner: <plugin>` to bind a dataset-specific training loop, "
        f"or use `plasformers.training.train_one_epoch` inside your project runner. Snapshot: {run.path}",
        style="green",
    )


@app.command()
def val(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint"),
    set_: List[str] = typer.Option([], "--set"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Validate a checkpoint through a registered evaluator."""

    bundle = _bundle(config, set_)
    evaluator = bundle.data.get("evaluator")
    if evaluator:
        result = GLOBAL_PLUGINS.get("evaluator", str(evaluator))(bundle.data, checkpoint)
    else:
        result = {
            "status": "ready",
            "message": "Register an evaluator plugin for COCO/VOC/OpenImages/custom validation.",
            "checkpoint": str(checkpoint) if checkpoint else None,
        }
    _write_json(result, output)


@app.command()
def predict(
    source: Path = typer.Argument(..., help="Image path."),
    variant: str = typer.Option("nano"),
    checkpoint: Optional[Path] = typer.Option(None),
    device: str = typer.Option("auto"),
    conf: float = typer.Option(0.25),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Run image inference."""

    from .engine.inference import InferenceEngine

    with status("Running inference..."):
        engine = InferenceEngine(
            variant=variant,
            checkpoint=checkpoint,
            device=device,
            conf_threshold=conf,
        )
        result = engine.predict(source)
    payload = {
        "fps": result.fps,
        "latency_ms": result.latency_ms,
        "detections": [asdict(det) for det in result.detections],
    }
    _write_json(payload, output)


@app.command()
def export(
    backend: str = typer.Argument(..., help="onnx, tensorrt, openvino, coreml, tflite, torchscript, tvm, rknn, ncnn, edgetpu"),
    variant: str = typer.Option("base"),
    checkpoint: Optional[Path] = typer.Option(None),
    output: Path = typer.Option(Path("exports/plasformers")),
    num_classes: int = typer.Option(80),
    height: int = typer.Option(640),
    width: int = typer.Option(640),
    batch: int = typer.Option(1),
    int8: bool = typer.Option(False),
    no_validate: bool = typer.Option(False),
) -> None:
    """Export a model to deployment backends with validation."""

    from .engine.exporters import ExportRequest, export_model

    request = ExportRequest(
        backend=backend,
        variant=variant,
        checkpoint=str(checkpoint) if checkpoint else None,
        output=output,
        num_classes=num_classes,
        input_shape=(batch, 3, height, width),
        int8=int8,
        validate=not no_validate,
    )
    result = export_model(request)
    print_panel("Export", json.dumps({"artifact": str(result.artifact), "validated": result.validated}, indent=2))


@app.command()
def benchmark(
    variant: str = typer.Option("nano"),
    all_: bool = typer.Option(False, "--all", help="Benchmark every model family member."),
    device: str = typer.Option("auto"),
    batch: int = typer.Option(1),
    height: int = typer.Option(640),
    width: int = typer.Option(640),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Measure FPS, latency, throughput, memory, and params."""

    from .engine.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(device=device)
    if all_:
        rows = runner.run_all(batch=batch, height=height, width=width)
    else:
        rows = [asdict(runner.run(variant, batch=batch, height=height, width=width))]
    print_table("Benchmark", rows)
    if output:
        _write_json(rows, output)


@app.command()
def profile(
    variant: str = typer.Option("base"),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """Profile model parameters and buffers."""

    from .engine.profile import profile_model

    record = profile_model(variant, output)
    _write_json(record, None)


@app.command()
def tune(
    trials: int = typer.Option(10),
    method: str = typer.Option("random", help="random, evolutionary, optuna."),
    output: Path = typer.Option(Path("runs/tune/results.json")),
) -> None:
    """Run AutoML search over learning rate, augmentations, losses, and scaling factors."""

    from .tune import HyperparameterTuner, default_detection_space

    def objective(params: dict[str, object]) -> float:
        scorer = GLOBAL_PLUGINS.get("tune_objective", "default") if any(GLOBAL_PLUGINS.list("tune_objective")) else None
        if scorer is not None:
            return float(scorer(params))
        return 0.0

    tuner = HyperparameterTuner(objective, default_detection_space())
    if method == "optuna":
        results = tuner.optuna_search(trials)
    elif method == "evolutionary":
        results = tuner.evolutionary_search(trials)
    else:
        results = tuner.random_search(trials)
    tuner.save_results(results, output)
    print_panel("Tune", f"Wrote {len(results)} trial records to {output}")


@app.command()
def prune(
    variant: str = typer.Option("base"),
    checkpoint: Optional[Path] = typer.Option(None),
    amount: float = typer.Option(0.2),
    output: Path = typer.Option(Path("exports/pruned.pt")),
) -> None:
    """Apply structured pruning and save a compressed checkpoint."""

    import torch
    from plasformers import build_plasformers
    from plasformers.export import load_checkpoint
    from .compression.pruning import structured_prune

    model = build_plasformers(variant, export_mode=True)
    if checkpoint:
        load_checkpoint(model, checkpoint)
    structured_prune(model, amount=amount)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "compression": {"prune_amount": amount}}, output)
    print_panel("Prune", f"Saved pruned checkpoint to {output}")


@app.command()
def quantize(
    variant: str = typer.Option("nano"),
    backend: str = typer.Option("fbgemm"),
    qat: bool = typer.Option(True),
    output: Path = typer.Option(Path("exports/qat_ready.pt")),
) -> None:
    """Prepare QAT or run plugin-backed post-training quantization."""

    import torch
    from plasformers import build_plasformers
    from .compression.quantization import prepare_quantization_aware_training

    model = build_plasformers(variant, export_mode=True)
    if qat:
        model = prepare_quantization_aware_training(model, backend=backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "quantization": {"backend": backend, "qat": qat}}, output)
    print_panel("Quantize", f"Saved quantization-ready checkpoint to {output}")


@app.command()
def distill(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    teacher: str = typer.Option("custom", help="YOLOv10, RT-DETR, or custom teacher identifier."),
    set_: List[str] = typer.Option([], "--set"),
    dry_run: bool = typer.Option(True),
) -> None:
    """Run teacher-student, feature, logit, relational, or self-distillation."""

    bundle = _bundle(config, set_)
    run = ExperimentRun(str(bundle.data.get("experiment", "plas_distill")))
    run.snapshot({**bundle.data, "teacher": teacher})
    if dry_run:
        print_panel("Distill", f"Distillation snapshot written to {run.path}")
        return
    runner = GLOBAL_PLUGINS.get("distill_runner", str(bundle.data.get("runner", "default")))
    runner(bundle.data, teacher, run)


@app.command()
def serve(
    variant: str = typer.Option("nano"),
    checkpoint: Optional[Path] = typer.Option(None),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    token: Optional[str] = typer.Option(None, help="Bearer token for /predict."),
) -> None:
    """Serve REST and WebSocket inference with FastAPI."""

    import uvicorn
    from .serve.api import create_app

    uvicorn.run(create_app(variant=variant, checkpoint=str(checkpoint) if checkpoint else None, auth_token=token), host=host, port=port)


@app.command()
def track(
    source: str = typer.Argument(..., help="Video path, stream URL, or camera index."),
    variant: str = typer.Option("nano"),
    checkpoint: Optional[Path] = typer.Option(None),
    tracker: str = typer.Option("bytetrack"),
    output: Path = typer.Option(Path("runs/track/tracks.jsonl")),
) -> None:
    """Run detection plus multi-object tracking."""

    from .engine.inference import InferenceEngine
    from .tracking import TRACKER_REGISTRY

    source_value: str | int = int(source) if source.isdigit() else source
    engine = InferenceEngine(variant=variant, checkpoint=checkpoint)
    mot = TRACKER_REGISTRY[tracker]()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for frame_id, result in enumerate(engine.stream_video(source_value)):
            tracks = mot.update(result.detections)
            handle.write(json.dumps({"frame": frame_id, "tracks": [asdict(track) for track in tracks]}) + "\n")
    print_panel("Track", f"Wrote tracks to {output}")


@app.command()
def explain(
    image: Path = typer.Argument(...),
    variant: str = typer.Option("nano"),
    target_layer: str = typer.Option("backbone.stage_modules.4"),
    output: Path = typer.Option(Path("runs/explain/heatmap.pt")),
) -> None:
    """Generate Grad-CAM style explanations."""

    import torch
    from .engine.inference import _preprocess
    from .explain import GradCAM
    from plasformers import build_plasformers

    model = build_plasformers(variant).eval()
    tensor, _ = _preprocess([image], (640, 640), "cpu", False)
    cam = GradCAM(model, target_layer)(tensor)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"heatmap": cam.heatmap, "method": cam.method, "target_layer": cam.target_layer}, output)
    print_panel("Explain", f"Saved explanation to {output}")


@app.command()
def doctor(output: Optional[Path] = typer.Option(None)) -> None:
    """Check dependencies, accelerators, tools, and security posture."""

    report = run_doctor()
    print_table("Packages", [{"package": k, "available": v} for k, v in report["packages"].items()])
    print_table("Tools", [{"tool": k, "available": v} for k, v in report["tools"].items()])
    if report["issues"]:
        print_panel("Issues", "\n".join(report["issues"]), style="yellow")
    if output:
        _write_json(report, output)


@app.command()
def tasks(output: Optional[Path] = typer.Option(None)) -> None:
    """List universal foundation vision tasks."""

    from .foundation.tasks import get_default_task_specs

    rows = [
        {"task": spec.task.value, "num_classes": spec.num_classes, "output_dim": spec.output_dim, "stride": spec.stride}
        for spec in get_default_task_specs().values()
    ]
    print_table("Foundation Tasks", rows)
    if output:
        _write_json(rows, output)


@app.command()
def hardware(output: Optional[Path] = typer.Option(None)) -> None:
    """Auto-detect hardware and recommended runtimes."""

    from dataclasses import asdict
    from .runtime.hardware import detect_hardware, optimal_runtime

    profile = detect_hardware()
    payload = {**asdict(profile), "optimal_runtime": optimal_runtime(profile)}
    _write_json(payload, output)


@app.command()
def registry(
    action: str = typer.Argument("list", help="list, register, promote, rollback"),
    name: str = typer.Option("plasformers"),
    version: str = typer.Option("0.1.0"),
    artifact: Optional[Path] = typer.Option(None),
    stage: str = typer.Option("dev"),
    root: Path = typer.Option(Path("registry")),
) -> None:
    """Manage enterprise model registry entries."""

    from dataclasses import asdict
    from .enterprise.registry import EnterpriseModelRegistry

    reg = EnterpriseModelRegistry(root)
    if action == "register":
        if artifact is None:
            raise typer.BadParameter("--artifact is required for register")
        entry = reg.register(name, version, artifact)
        _write_json(asdict(entry), None)
    elif action == "promote":
        _write_json(asdict(reg.promote(name, version, stage)), None)
    elif action == "rollback":
        _write_json(asdict(reg.rollback(name, stage)), None)
    else:
        _write_json([asdict(entry) for entry in reg.list(name if name else None)], None)


@app.command()
def marketplace(
    query: str = typer.Argument("", help="Search query."),
    kind: Optional[str] = typer.Option(None),
    index: Path = typer.Option(Path("marketplace/index.json")),
) -> None:
    """Search the extension marketplace."""

    from dataclasses import asdict
    from .ecosystem.marketplace import MarketplaceIndex

    rows = [asdict(item) for item in MarketplaceIndex(index).search(query, kind)]
    print_table("Marketplace", rows)


@app.command()
def arena(output: Path = typer.Option(Path("benchmarks/arena_results.json"))) -> None:
    """Run benchmark arena command scaffolds for external baselines."""

    from dataclasses import asdict
    from .ecosystem.arena import BenchmarkArena

    results = BenchmarkArena().run(output)
    print_table("Benchmark Arena", [asdict(result) for result in results])


@app.command()
def copilot(
    metrics: Optional[Path] = typer.Option(None, help="JSON metrics file."),
    baseline: Optional[Path] = typer.Option(None, help="JSON baseline metrics file."),
    output: Path = typer.Option(Path("runs/copilot/summary.json")),
) -> None:
    """Generate ablations, regression checks, summaries, and next experiments."""

    from dataclasses import asdict
    from .ecosystem.assistant import ResearchCopilot

    current = json.loads(metrics.read_text(encoding="utf-8")) if metrics else {}
    base = json.loads(baseline.read_text(encoding="utf-8")) if baseline else {}
    summary = ResearchCopilot().summarize(current, base)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print_panel("Research Copilot", f"Wrote summary to {output}")


@app.command()
def web(
    variant: str = typer.Option("nano"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
) -> None:
    """Launch the interactive Gradio interface."""

    from .apps.web import launch

    launch(default_variant=variant, host=host, port=port)


@app.command()
def zoo(
    download: Optional[str] = typer.Option(None, help="Download a registered checkpoint by name."),
    metadata: Optional[Path] = typer.Option(None, help="Write model-zoo metadata JSON."),
) -> None:
    """List or download model-zoo checkpoints."""

    registry = get_model_zoo()
    if download:
        path = registry.download(download)
        print_panel("Model Zoo", f"Downloaded {download} to {path}")
        return
    if metadata:
        registry.export_metadata(metadata)
    print_table("Model Zoo", registry.as_rows())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
