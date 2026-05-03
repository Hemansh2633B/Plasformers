import pytest
from pathlib import Path
from tests.synthetic.generators import generate_coco_json, generate_yolo_dataset
import os

@pytest.mark.integration
def test_yolo_dataset_generation(tmp_path):
    dataset_path = tmp_path / "yolo_data"
    generate_yolo_dataset(dataset_path, num_images=3, num_classes=5)

    assert (dataset_path / "images").exists()
    assert (dataset_path / "labels").exists()
    assert (dataset_path / "dataset.yaml").exists()
    assert len(list((dataset_path / "images").glob("*.jpg"))) == 3

@pytest.mark.integration
def test_coco_json_generation(tmp_path):
    coco_path = tmp_path / "coco.json"
    generate_coco_json(coco_path, num_images=2, num_classes=10)

    import json
    with open(coco_path) as f:
        data = json.load(f)

    assert len(data["images"]) == 2
    assert len(data["categories"]) == 10
    assert "annotations" in data

@pytest.mark.integration
def test_dataset_pipeline_validation():
    # Test dataset format validation logic if it exists in plas/data
    pass
