"""Persist one prediction and pre-NMS audit record per evaluated image."""
import json

from .coco_split import digest, write_json
from .unified_infer import Runner, decode_image, letterbox, postprocess


def read_records(path, expected_ids):
    marker=path.with_suffix(".provenance.json")
    stamped=False
    if marker.is_file():
        provenance=json.loads(marker.read_text(encoding="utf-8"))
        stamped="records_sha256" in provenance
        if stamped and provenance["records_sha256"]!=digest(path):
            raise ValueError(f"Prediction SHA-256 differs from saved manifest: {path.name}")
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            image = obj.pop("image_id")
            if image in records:
                raise ValueError(f"Duplicate image {image} in {path}")
            records[image] = obj
    if set(records) != set(expected_ids) or (stamped and list(records)!=list(expected_ids)):
        raise ValueError(f"Prediction coverage mismatch: {path}")
    return records


def save_records(path, resolution, backend, device, ids, images, categories, config, provenance):
    metadata = path.with_suffix(".provenance.json")
    if path.exists() or metadata.exists():
        if not path.is_file() or not metadata.is_file():
            raise ValueError(f"Incomplete prediction cache for {path.name}; choose a fresh output directory")
        saved=json.loads(metadata.read_text(encoding="utf-8"))
        checksum=saved.pop("records_sha256",None)
        if saved!=provenance or checksum!=digest(path):
            raise ValueError(f"Prediction cache or manifest differs for {path.name}; choose a fresh output directory")
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
    write_json(metadata, {**provenance, "records_sha256":digest(path)})
    print(f"Saved {path.name}: {len(ids)} images", flush=True)
