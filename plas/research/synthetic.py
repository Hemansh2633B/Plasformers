"""Synthetic data factory with procedural, Blender, Unreal, and physics hooks."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ObjectAsset:
    """Renderable object asset."""

    name: str
    mesh: str | None = None
    texture: str | None = None
    class_id: int = 0


@dataclass
class SceneSpec:
    """Procedural scene generation specification."""

    width: int = 1024
    height: int = 1024
    objects: list[ObjectAsset] = field(default_factory=list)
    lighting: dict[str, Any] = field(default_factory=lambda: {"type": "random_hdr"})
    camera: dict[str, Any] = field(default_factory=lambda: {"fov": [35, 90], "pose": "random"})
    physics: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    domain_randomization: dict[str, Any] = field(default_factory=lambda: {"textures": True, "weather": True, "motion_blur": True})


class SyntheticDataFactory:
    """Generate manifests and invoke external simulators for synthetic datasets."""

    def __init__(self, output_dir: str | Path = "synthetic") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_scene(self, assets: Iterable[ObjectAsset], max_objects: int = 12) -> SceneSpec:
        pool = list(assets)
        count = random.randint(1, max(1, min(max_objects, len(pool))))
        return SceneSpec(objects=random.sample(pool, count))

    def auto_label(self, scene: SceneSpec) -> dict[str, Any]:
        """Create deterministic placeholder labels from scene object placements."""

        annotations = []
        for idx, asset in enumerate(scene.objects, start=1):
            x = (idx * 37) % max(1, scene.width - 64)
            y = (idx * 53) % max(1, scene.height - 64)
            annotations.append({"id": idx, "category_id": asset.class_id, "bbox": [x, y, 64, 64], "asset": asset.name})
        return {"scene": asdict(scene), "annotations": annotations}

    def write_manifest(self, scenes: Iterable[SceneSpec], name: str = "manifest.jsonl") -> Path:
        out = self.output_dir / name
        with out.open("w", encoding="utf-8") as handle:
            for scene in scenes:
                handle.write(json.dumps(self.auto_label(scene)) + "\n")
        return out

    def blender_command(self, scene_file: str | Path, script: str | Path) -> list[str]:
        return ["blender", "--background", "--python", str(script), "--", str(scene_file), str(self.output_dir)]

    def run_blender(self, scene_file: str | Path, script: str | Path) -> None:
        subprocess.run(self.blender_command(scene_file, script), check=True)

    def unreal_export_manifest(self, scenes: Iterable[SceneSpec], output: str | Path) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(scene) for scene in scenes], indent=2), encoding="utf-8")
        return out

    def physics_hook(self, scene: SceneSpec, simulator: Any) -> Mapping[str, Any]:
        """Call a user-provided physics simulator hook."""

        return simulator(asdict(scene))
