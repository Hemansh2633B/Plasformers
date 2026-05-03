"""Academic toolkit for reproductions, citations, appendices, and LaTeX exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .arena import latex_benchmark_table


@dataclass(frozen=True)
class Citation:
    """BibTeX citation metadata."""

    key: str
    title: str
    author: str
    year: int
    url: str

    def bibtex(self) -> str:
        return (
            f"@misc{{{self.key},\n"
            f"  title={{{self.title}}},\n"
            f"  author={{{self.author}}},\n"
            f"  year={{{self.year}}},\n"
            f"  url={{{self.url}}}\n"
            "}"
        )


def reproduction_pipeline(config: str | Path, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"config": str(config), "steps": ["train", "val", "export", "benchmark", "appendix"]}, indent=2),
        encoding="utf-8",
    )
    return out


def appendix(metrics: Mapping[str, object], ablations: Iterable[Mapping[str, object]]) -> str:
    return "\n\n".join(
        [
            "# Appendix",
            "## Metrics",
            json.dumps(dict(metrics), indent=2),
            "## Ablations",
            json.dumps([dict(item) for item in ablations], indent=2),
        ]
    )


def write_latex_table(rows: Iterable[Mapping[str, object]], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex_benchmark_table(rows), encoding="utf-8")
    return out
