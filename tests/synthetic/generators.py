import numpy as np
import torch
from pathlib import Path
import json
import cv2
from typing import Tuple, List, Dict, Any

def generate_mock_image(height: int = 640, width: int = 640) -> np.ndarray:
    """Generates a random noise image."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def generate_coco_json(output_path: Path, num_images: int = 5, num_classes: int = 80):
    """Generates a minimal COCO format JSON."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(num_classes)]
    }

    ann_id = 1
    for i in range(num_images):
        img_id = i + 1
        coco["images"].append({
            "id": img_id,
            "file_name": f"img_{img_id}.jpg",
            "height": 640,
            "width": 640
        })

        # Add 1-3 random bboxes
        for _ in range(np.random.randint(1, 4)):
            x, y = np.random.randint(0, 400, 2)
            w, h = np.random.randint(10, 200, 2)
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": np.random.randint(0, num_classes),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    with open(output_path, "w") as f:
        json.dump(coco, f)

def generate_yolo_dataset(root_path: Path, num_images: int = 5, num_classes: int = 80):
    """Generates a minimal YOLO format dataset."""
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "images").mkdir(exist_ok=True)
    (root_path / "labels").mkdir(exist_ok=True)

    for i in range(num_images):
        img_name = f"img_{i}.jpg"
        img = generate_mock_image()
        cv2.imwrite(str(root_path / "images" / img_name), img)

        with open(root_path / "labels" / f"img_{i}.txt", "w") as f:
            for _ in range(np.random.randint(1, 4)):
                cls = np.random.randint(0, num_classes)
                cx, cy, w, h = np.random.rand(4)
                f.write(f"{cls} {cx} {cy} {w} {h}\n")

    with open(root_path / "dataset.yaml", "w") as f:
        f.write(f"path: {root_path.absolute()}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {[f'class_{i}' for i in range(num_classes)]}\n")

def create_synthetic_dataloader(batch_size: int = 2, img_size: int = 640):
    """Creates a generator that yields mock batches (images, targets)."""
    while True:
        images = torch.randn(batch_size, 3, img_size, img_size)
        # Mock targets: [batch_idx, cls, cx, cy, w, h]
        targets = torch.zeros((batch_size * 3, 6))
        for i in range(batch_size):
            targets[i*3:(i+1)*3, 0] = i
            targets[i*3:(i+1)*3, 1] = torch.randint(0, 80, (3,))
            targets[i*3:(i+1)*3, 2:] = torch.rand((3, 4))
        yield images, targets
