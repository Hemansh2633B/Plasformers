"""Minimal Python inference example."""

from __future__ import annotations

from plas.engine.inference import InferenceEngine


def main() -> None:
    engine = InferenceEngine(variant="nano", device="auto")
    result = engine.predict("examples/assets/sample.jpg")
    print(result)


if __name__ == "__main__":
    main()
