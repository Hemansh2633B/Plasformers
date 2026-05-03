"""Dataset management tests."""

from __future__ import annotations

import json
from pathlib import Path

from plas.data.manager import DatasetManager


def test_validate_coco(tmp_path: Path) -> None:
    ann = tmp_path / "ann.json"
    ann.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg"}],
                "categories": [{"id": 1, "name": "thing"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
            }
        ),
        encoding="utf-8",
    )
    report = DatasetManager(tmp_path).validate_coco(ann)
    assert report.images == 1
    assert report.annotations == 1
    assert report.classes["thing"] == 1
