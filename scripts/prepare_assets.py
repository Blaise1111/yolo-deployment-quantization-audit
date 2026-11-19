"""Download and extract the COCO validation distribution and official YOLO11n weights."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coco_split import ANNOTATIONS, IMAGES, ROOT, WEIGHTS, digest, load_config

ASSETS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def download(item):
    filename, url = item
    path = ROOT / "data" / filename
    if not path.is_file():
        piece = path.with_suffix(path.suffix + ".part")
        with (ROOT / "logs" / f"download_{filename}.log").open("ab") as log:
            subprocess.run(["curl", "-fLsS", "--retry", "4", "--connect-timeout", "20",
                            "-C", "-", "-o", str(piece), url],
                           stdout=log, stderr=log, check=True)
        piece.replace(path)
    return path


if __name__ == "__main__":
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(3) as pool:
        val, annotations, _ = pool.map(download, ASSETS.items())
    if digest(WEIGHTS) != load_config()["weight_sha256"]:
        raise ValueError("Official YOLO11n checkpoint SHA-256 mismatch")
    if not ANNOTATIONS.is_file():
        with ZipFile(annotations) as source:
            source.extract("annotations/instances_val2017.json", ROOT / "data")
    with ZipFile(val) as source:
        members = [n for n in source.namelist() if n.startswith("val2017/") and n.endswith(".jpg")]
        if len(members) != 5000:
            raise ValueError(f"COCO archive contained {len(members)} images")
        for name in members:
            target = IMAGES / Path(name).name
            if not target.is_file() or target.stat().st_size != source.getinfo(name).file_size:
                source.extract(name, ROOT / "data")
    for path in val, annotations, WEIGHTS:
        print(f"{path.name} SHA-256 {digest(path)}")
