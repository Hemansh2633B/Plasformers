"""Export Plasformers to ONNX."""

from __future__ import annotations

import argparse

from plasformers.export import build_export_model, export_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Plasformers to ONNX")
    parser.add_argument("--variant", default="base", choices=["nano", "small", "base", "large", "xlarge"])
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="exports/plasformers.onnx")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static", action="store_true", help="Disable dynamic axes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_export_model(args.variant, args.num_classes, args.checkpoint)
    export_onnx(
        model,
        args.output,
        input_shape=(args.batch, 3, args.height, args.width),
        opset=args.opset,
        dynamic=not args.static,
    )


if __name__ == "__main__":
    main()
