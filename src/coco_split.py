"""Frozen COCO partitions and evaluation ground truth."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data/yolo11n.pt"
ANNOTATIONS = ROOT / "data/annotations/instances_val2017.json"
IMAGES = ROOT / "data/val2017"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config():
    return yaml.safe_load((ROOT / "configs/coco_audit.yaml").read_text(encoding="utf-8"))


def split_ids(ids, seed, calib_count=200):
    ids = sorted(ids)
    if len(ids) != 5000 or len(set(ids)) != 5000 or calib_count != 200:
        raise ValueError("Expected 5,000 unique official image IDs and 200 calibration images")
    permutation = np.random.default_rng(seed).permutation(ids).tolist()
    calibration = sorted(permutation[:calib_count])
    evaluation = sorted(permutation[calib_count:])
    if set(calibration) & set(evaluation):
        raise RuntimeError("COCO calibration and evaluation overlap")
    return calibration, evaluation, sorted(permutation[calib_count:calib_count + 1000])


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def coco_inputs(config):
    if digest(WEIGHTS) != config["weight_sha256"]:
        raise ValueError("Checkpoint SHA-256 fails verification")
    source = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    images = {row["id"]: row for row in source["images"]}
    calibration, evaluation, fast = split_ids(images, config["seed"], config["calib_count"])
    for row in images.values():
        if not (IMAGES / row["file_name"]).is_file():
            raise FileNotFoundError(row["file_name"])
    partitions = dict(calib_200=calibration, eval_4800=evaluation, eval_1000=fast)
    return source, images, partitions


def ground_truth(source, ids):
    ids = set(ids)
    return dict(
        images=[row for row in source["images"] if row["id"] in ids],
        annotations=[{key: row[key] for key in ("id", "image_id", "category_id", "bbox", "area", "iscrowd")}
                     for row in source["annotations"] if row["image_id"] in ids],
        categories=source["categories"],
    )
