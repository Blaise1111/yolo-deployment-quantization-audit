"""Persist one prediction and pre-NMS audit record per evaluated image."""
import json
from pathlib import Path

from .coco_split import write_json
from .unified_infer import Runner, decode_image, letterbox, postprocess


def read_records(path, expected_ids):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            image = obj.pop("image_id")
            if image in records:
                raise ValueError(f"Duplicate image {image} in {path}")
            records[image] = obj
    if set(records) != set(expected_ids):
        raise ValueError(f"Prediction coverage mismatch: {path}")
    return records


def save_records(path, resolution, backend, device, ids, images, categories, config, provenance):
    metadata = path.with_suffix(".provenance.json")
    if path.is_file() and metadata.is_file() and json.loads(metadata.read_text()) == provenance:
        return
    temporary = path.with_suffix(".part")
    runner = Runner(resolution, backend, device, config["cpu_threads"])
    try:
        if runner.names and [runner.names[i] for i in range(len(categories))] != [
                item["name"] for item in categories]:
            raise ValueError("Model class order does not match sorted COCO categories")
        category_ids = [item["id"] for item in categories]
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for index, image_id in enumerate(ids, 1):
                image = decode_image(images[image_id]["file_name"])
                tensor, geometry = letterbox(image, resolution)
                record = postprocess(runner(tensor), geometry, category_ids, config,
                                     record_candidates=not path.name.startswith("calib_"))
                stream.write(json.dumps({"image_id": image_id, **record}, separators=(",", ":"),
                                        allow_nan=False) + "\n")
                if index % 1000 == 0:
                    print(f"{path.stem}: {index}/{len(ids)}", flush=True)
    finally:
        runner.close()
    temporary.replace(path)
    write_json(metadata, provenance)
    print(f"Saved {path.name}: {len(ids)} images", flush=True)
