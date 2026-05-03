"""Developer-experience helpers for graph exploration, debugging, and live profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_model_graph(model: Any, output: str | Path) -> Path:
    """Export a simple module graph as JSON."""

    nodes = []
    edges = []
    previous = None
    for name, module in model.named_modules():
        if not name:
            continue
        nodes.append({"id": name, "type": module.__class__.__name__})
        parent = ".".join(name.split(".")[:-1])
        if parent:
            edges.append({"source": parent, "target": name})
        elif previous:
            edges.append({"source": previous, "target": name})
        previous = name
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2), encoding="utf-8")
    return out


def vscode_extension_manifest() -> dict[str, object]:
    return {
        "name": "plasformers-tools",
        "displayName": "Plasformers Tools",
        "contributes": {
            "commands": [
                {"command": "plasformers.profile", "title": "Plasformers: Profile Model"},
                {"command": "plasformers.graph", "title": "Plasformers: Open Graph Explorer"},
            ]
        },
    }


def live_profile_dashboard_spec(port: int = 8050) -> dict[str, object]:
    return {"server": "dash-or-streamlit", "port": port, "panels": ["latency", "memory", "throughput", "gpu_utilization"]}
