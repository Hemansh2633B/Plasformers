"""Dataset download, validation, conversion, and data-centric utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping
from xml.etree import ElementTree


@dataclass
class DatasetReport:
    """Dataset quality report."""

    images: int = 0
    annotations: int = 0
    classes: dict[str, int] = field(default_factory=dict)
    missing_images: list[str] = field(default_factory=list)
    duplicate_files: list[list[str]] = field(default_factory=list)
    invalid_boxes: int = 0


class DatasetManager:
    """Manage common detection dataset workflows."""

    def __init__(self, root: str | Path = "datasets") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self, name: str, url: str, sha256: str | None = None) -> Path:
        out = self.root / name
        archive = self.root / f"{name}.download"
        urllib.request.urlretrieve(url, archive)
        if sha256 and self._sha256(archive) != sha256.lower():
            archive.unlink(missing_ok=True)
            raise RuntimeError(f"Dataset download failed SHA256 validation: {name}")
        out.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out)
        else:
            shutil.move(str(archive), str(out / archive.name))
        return out

    def validate_coco(self, annotation_file: str | Path, image_dir: str | Path | None = None) -> DatasetReport:
        data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        images = {item["id"]: item for item in data.get("images", [])}
        categories = {item["id"]: item["name"] for item in data.get("categories", [])}
        report = DatasetReport(images=len(images), annotations=len(data.get("annotations", [])))
        for ann in data.get("annotations", []):
            name = categories.get(ann.get("category_id"), str(ann.get("category_id")))
            report.classes[name] = report.classes.get(name, 0) + 1
            box = ann.get("bbox", [0, 0, 0, 0])
            if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                report.invalid_boxes += 1
        if image_dir:
            root = Path(image_dir)
            for image in images.values():
                if not (root / image["file_name"]).exists():
                    report.missing_images.append(image["file_name"])
        return report

    def find_duplicates(self, directory: str | Path, pattern: str = "*") -> list[list[str]]:
        buckets: Dict[str, list[str]] = {}
        for path in Path(directory).rglob(pattern):
            if path.is_file():
                buckets.setdefault(self._sha256(path), []).append(str(path))
        return [items for items in buckets.values() if len(items) > 1]

    def class_balance(self, labels: Iterable[int]) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
        return counts

    def hard_example_manifest(self, examples: Iterable[Mapping[str, object]], output: str | Path) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for item in examples:
                handle.write(json.dumps(dict(item)) + "\n")
        return out

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def convert_yolo_to_coco(labels_dir: str | Path, image_dir: str | Path, output: str | Path) -> Path:
    """Convert simple YOLO txt labels to COCO JSON."""

    images = []
    annotations = []
    ann_id = 1
    for image_id, image_path in enumerate(sorted(Path(image_dir).glob("*")), start=1):
        if not image_path.is_file():
            continue
        images.append({"id": image_id, "file_name": image_path.name})
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls),
                    "bbox": [cx - w / 2, cy - h / 2, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"images": images, "annotations": annotations, "categories": []}), encoding="utf-8")
    return out


def convert_voc_to_coco(annotations_dir: str | Path, output: str | Path) -> Path:
    """Convert Pascal VOC XML annotations to COCO JSON."""

    images = []
    annotations = []
    categories: dict[str, int] = {}
    ann_id = 1
    for image_id, xml_path in enumerate(sorted(Path(annotations_dir).glob("*.xml")), start=1):
        root = ElementTree.parse(xml_path).getroot()
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        images.append({"id": image_id, "file_name": filename})
        for obj in root.findall("object"):
            name = obj.findtext("name") or "object"
            categories.setdefault(name, len(categories) + 1)
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": categories[name],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for name, idx in categories.items()],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def convert_openimages_to_coco(csv_file: str | Path, output: str | Path) -> Path:
    """Convert an OpenImages detection CSV subset to COCO JSON."""

    import csv

    images: dict[str, int] = {}
    categories: dict[str, int] = {}
    annotations = []
    ann_id = 1
    with Path(csv_file).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = row.get("ImageID", "unknown")
            images.setdefault(image_name, len(images) + 1)
            label = row.get("LabelName", "object")
            categories.setdefault(label, len(categories) + 1)
            xmin = float(row.get("XMin", 0.0))
            xmax = float(row.get("XMax", 0.0))
            ymin = float(row.get("YMin", 0.0))
            ymax = float(row.get("YMax", 0.0))
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": images[image_name],
                    "category_id": categories[label],
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    payload = {
        "images": [{"id": idx, "file_name": name} for name, idx in images.items()],
        "annotations": annotations,
        "categories": [{"id": idx, "name": name} for name, idx in categories.items()],
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def convert_dataset(source_format: str, source: str | Path, output: str | Path, **kwargs: object) -> Path:
    """Dispatch dataset format conversion to COCO JSON."""

    fmt = source_format.lower()
    if fmt == "coco":
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out)
        return out
    if fmt == "yolo":
        image_dir = kwargs.get("image_dir")
        if image_dir is None:
            raise ValueError("YOLO conversion requires image_dir=<path>")
        return convert_yolo_to_coco(source, image_dir, output)
    if fmt == "voc":
        return convert_voc_to_coco(source, output)
    if fmt == "openimages":
        return convert_openimages_to_coco(source, output)
    if fmt == "custom":
        converter = kwargs.get("converter")
        if converter is None:
            raise ValueError("Custom conversion requires converter callable")
        return converter(source, output)
    raise ValueError(f"Unsupported dataset format: {source_format}")
