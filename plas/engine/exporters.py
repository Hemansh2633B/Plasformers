"""Unified export backend dispatcher."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

from plasformers.export import build_export_model, export_onnx, trace_torchscript


@dataclass
class ExportRequest:
    """Description of a model export job."""

    backend: str
    variant: str = "base"
    checkpoint: str | None = None
    num_classes: int = 80
    output: str | Path = "exports/plasformers"
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640)
    opset: int = 17
    fp16: bool = True
    int8: bool = False
    validate: bool = True
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportResult:
    """Result of an export job."""

    backend: str
    artifact: Path
    validated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _run_command(args: Sequence[str]) -> None:
    subprocess.run(list(args), check=True)


def _validate_onnx(path: Path) -> bool:
    try:
        import onnx
    except ImportError:
        return False
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return True


def export_model(request: ExportRequest) -> ExportResult:
    """Export a model to a supported backend.

    Backends that depend on vendor toolchains delegate to the vendor CLI when
    installed and return actionable errors otherwise.
    """

    backend = request.backend.lower()
    output = Path(request.output)
    model = build_export_model(request.variant, request.num_classes, request.checkpoint)

    if backend == "onnx":
        artifact = output.with_suffix(".onnx")
        export_onnx(model, artifact, request.input_shape, request.opset, dynamic=True)
        return ExportResult(backend, artifact, _validate_onnx(artifact) if request.validate else False)

    if backend == "torchscript":
        artifact = output.with_suffix(".torchscript.pt")
        trace_torchscript(model, artifact, request.input_shape)
        return ExportResult(backend, artifact, artifact.exists())

    if backend == "tensorrt":
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        trtexec = shutil.which("trtexec")
        if trtexec is None:
            raise RuntimeError("TensorRT export requires 'trtexec' on PATH")
        artifact = output.with_suffix(".engine")
        cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={artifact}"]
        if request.fp16:
            cmd.append("--fp16")
        if request.int8:
            cmd.append("--int8")
        _run_command(cmd)
        return ExportResult(backend, artifact, artifact.exists(), {"onnx": str(onnx_path)})

    if backend == "openvino":
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        ovc = shutil.which("ovc") or shutil.which("mo")
        if ovc is None:
            raise RuntimeError("OpenVINO export requires 'ovc' or 'mo' on PATH")
        out_dir = output.parent / f"{output.stem}_openvino"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_command([ovc, str(onnx_path), "--output_model", str(out_dir / f"{output.stem}.xml")])
        return ExportResult(backend, out_dir, True, {"onnx": str(onnx_path)})

    if backend == "coreml":
        try:
            import coremltools as ct
        except ImportError as exc:
            raise RuntimeError("CoreML export requires coremltools") from exc
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        mlmodel = ct.converters.onnx.convert(model=str(onnx_path))
        artifact = output.with_suffix(".mlmodel")
        mlmodel.save(str(artifact))
        return ExportResult(backend, artifact, artifact.exists())

    if backend in {"tflite", "edgetpu"}:
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        try:
            from onnx_tf.backend import prepare
            import onnx
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TFLite export requires onnx, onnx-tf, and tensorflow") from exc
        tf_rep = prepare(onnx.load(str(onnx_path)))
        saved_model = output.parent / f"{output.stem}_saved_model"
        tf_rep.export_graph(str(saved_model))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
        artifact = output.with_suffix(".tflite")
        artifact.write_bytes(converter.convert())
        if backend == "edgetpu":
            compiler = shutil.which("edgetpu_compiler")
            if compiler is None:
                raise RuntimeError("Edge TPU compilation requires edgetpu_compiler")
            _run_command([compiler, str(artifact), "-o", str(output.parent)])
        return ExportResult(backend, artifact, artifact.exists())

    if backend == "ncnn":
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        tool = shutil.which("onnx2ncnn")
        if tool is None:
            raise RuntimeError("NCNN export requires onnx2ncnn on PATH")
        param = output.with_suffix(".param")
        bin_path = output.with_suffix(".bin")
        _run_command([tool, str(onnx_path), str(param), str(bin_path)])
        return ExportResult(backend, param, param.exists() and bin_path.exists(), {"bin": str(bin_path)})

    if backend in {"tvm", "rknn"}:
        onnx_path = output.with_suffix(".onnx")
        export_onnx(model, onnx_path, request.input_shape, request.opset, dynamic=False)
        raise RuntimeError(
            f"{backend.upper()} export requires target-specific calibration and SDK setup. "
            f"Generated ONNX seed artifact at {onnx_path}."
        )

    raise ValueError(f"Unsupported export backend: {request.backend}")
