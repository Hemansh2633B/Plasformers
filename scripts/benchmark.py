"""Simple PyTorch latency benchmark for Plasformers."""

from __future__ import annotations

import argparse
import time

import torch

from plasformers import build_plasformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Plasformers eager inference")
    parser.add_argument("--variant", default="nano", choices=["nano", "small", "base", "large", "xlarge"])
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--export-mode", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_plasformers(args.variant, export_mode=args.export_mode).to(device).eval()
    x = torch.randn(args.batch, 3, args.height, args.width, device=device)
    use_amp = args.amp and device == "cuda"

    for _ in range(args.warmup):
        with torch.autocast(device_type=device, enabled=use_amp):
            _ = model.export_forward(x) if args.export_mode else model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        with torch.autocast(device_type=device, enabled=use_amp):
            _ = model.export_forward(x) if args.export_mode else model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    images = args.batch * args.iters
    print(f"variant={args.variant} device={device} images/s={images / elapsed:.2f}")


if __name__ == "__main__":
    main()
