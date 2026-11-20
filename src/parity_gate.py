"""Bind a passing fast FP32 parity check to the exact evidence used by a full run."""
import json
from math import isfinite

from .coco_split import ANNOTATIONS, WEIGHTS, digest, write_json
from .export_quantize import image_fingerprints, validated_fp32_sha256

GATE_NAME = "fp32_parity_pass.json"
PARITY_LIMIT = 0.003
REBUILD = "Run --fast-eval-1000 --output-dir <fresh-fast-dir>, then pass that directory to --fast-gate-dir."


class ParityGateError(ValueError):
    pass


def parity_contract(config, partitions, images):
    checkpoint = digest(WEIGHTS)
    exports = {str(size): validated_fp32_sha256(size, config["cpu_threads"]) for size in config["resolutions"]}
    return dict(schema=1, checkpoint_sha256=checkpoint, fp32_sha256=exports,
                config=config, annotations_sha256=digest(ANNOTATIONS), partitions=partitions,
                fast_images=image_fingerprints(partitions["eval_1000"], images))


def valid_gaps(gaps, expected):
    return (isinstance(gaps, dict) and gaps.keys() == expected["fp32_sha256"].keys()
            and all(type(value) in (int, float) and isfinite(value) and 0 <= value <= PARITY_LIMIT
                    for value in gaps.values()))


def publish_gate(path, expected, gaps):
    if not valid_gaps(gaps, expected):
        raise ParityGateError(f"Fast FP32 gaps are incomplete or exceed {PARITY_LIMIT}. {REBUILD}")
    write_json(path, dict(contract=expected, abs_map_gap_by_resolution=gaps))


def read_gate(path):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ParityGateError(f"Missing or unreadable fast parity manifest. {REBUILD}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("contract"), dict):
        raise ParityGateError(f"Legacy or incomplete fast parity manifest. {REBUILD}")
    return record


def validate_gate(record, expected):
    actual = record["contract"]
    differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
    if differing:
        raise ParityGateError(f"Fast parity manifest mismatch ({', '.join(differing)}). {REBUILD}")
    if not valid_gaps(record.get("abs_map_gap_by_resolution"), expected):
        raise ParityGateError(f"Missing or failing fast FP32 gap evidence. {REBUILD}")
