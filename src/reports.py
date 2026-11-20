"""Rebuild all panels and numeric reports from saved predictions and timings."""
import json

import numpy as np
import pandas as pd

from .coco_metrics import evaluate, fixed_fppi
from .coco_split import split_ids
from .detections import read_records
from .flips_and_ece import audit_pair, ece_delta_ci, reliability, temperature_fit, transform
from .latency_bench import summarize_latency
from .parity_gate import PARITY_LIMIT
from .timing_evidence import load as load_timing, path_for, validate_rows

REPORTS = (
    "p2_backend_flips_audit", "p2_scale_degradation_matrix",
    "p2_ece_before_after_recal", "p2_latency_scaling_breakdown",
)


def csv_write(frame, path):
    temporary = path.with_suffix(".part")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    temporary.replace(path)


def detection_file(directory, size, backend):
    return directory / f"detections_{backend}_{size}.jsonl"


def cohort(status, config):
    active = status[status[:, 0] >= config["operating_confidence"]]
    return active[:, 0], (active[:, 1] > 0).astype(float), active[:, 2].astype(int)


def detection_reports(directory):
    metadata = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    parts = metadata["partitions"]
    partition_list = split_ids(parts["calib_200"] + parts["eval_4800"], config["seed"], config["calib_count"])
    if tuple(parts[key] for key in ("calib_200", "eval_4800", "eval_1000")) != partition_list:
        raise ValueError("Saved COCO partition must match the frozen seed")
    ids = parts[metadata["eval_partition"]]
    cal_ids = parts["calib_200"]
    gt = json.loads((directory/"eval_ground_truth.json").read_text(encoding="utf-8"))
    cal_gt = json.loads((directory/"calib_ground_truth.json").read_text(encoding="utf-8"))
    if sorted(row["id"] for row in gt["images"]) != ids or sorted(
            row["id"] for row in cal_gt["images"]) != cal_ids:
        raise ValueError("Ground truth and frozen partitions disagree")
    flips, ranking, calibration, residuals, events = [], [], [], {}, {}
    for size in config["resolutions"]:
        ref = read_records(detection_file(directory, size, "torch_fp32"), ids)
        for backend, device in metadata["accuracy_backends"].items():
            records = ref if backend == "torch_fp32" else read_records(
                detection_file(directory, size, backend), ids)
            common = dict(resolution=size, backend=backend, accuracy_device=device,
                          core=backend != "onnx_fp16", images=len(ids))
            metrics, status, anns, count = evaluate(gt, records, config["max_det"])
            ranking.append(dict(**common, **metrics, **fixed_fppi(status, anns, count, config["fppi"])))
            flip_result, delta, found = audit_pair(ref, records, config)
            flips.append(dict(**common, **flip_result))
            residuals[size, backend], events[size, backend] = delta, found
            scores, truth, image_ids = cohort(status, config)
            variants = [("raw", scores, None, None)]
            if backend == "onnx_int8":
                cal = read_records(directory / f"calib_int8_{size}.jsonl", cal_ids)
                _, observed, _, _ = evaluate(cal_gt, cal, config["max_det"], full=False)
                x, y, _ = cohort(observed, config)
                temperature = temperature_fit(x, y)
                adjusted = transform(scores, temperature)
                raw_all = np.unique([r[2] for record in records.values() for r in record["detections"]])
                if np.any(np.diff(transform(raw_all, temperature)) <= 0):
                    raise ValueError("Temperature scaling changed detection-score order")
                c = ece_delta_ci(scores, adjusted, truth, np.searchsorted(ids, image_ids), len(ids), config)
                variants.append(("temperature", adjusted, temperature, c))
            raw_ece = reliability(scores, truth, config["calibration_bins"])[0]["ece"]
            for method, values, temperature, ci in variants:
                result, bins = reliability(values, truth, config["calibration_bins"])
                calibration.append(dict(kind="metric", **common, method=method,
                                       temperature=temperature, delta_ece=result["ece"]-raw_ece,
                                       delta_ece_ci_low=ci[0] if ci is not None else None,
                                       delta_ece_ci_high=ci[1] if ci is not None else None, **result))
                calibration.extend(dict(kind="bin", **common, method=method, **item) for item in bins)
            print(f"Recomputed COCO and audit: {size} {backend} ({count} images)", flush=True)

    return {REPORTS[0]: pd.DataFrame(flips), REPORTS[1]: pd.DataFrame(ranking),
            REPORTS[2]: pd.DataFrame(calibration)}, (residuals, events)


def latency_report(directory):
    path = directory/"latency_logs.csv"
    if path_for(path).is_file():
        rows, _ = load_timing(path)
    else:
        print("Timing provenance: historical unauthenticated capture, offline recomputation only.", flush=True)
        rows = pd.read_csv(path, float_precision="round_trip")
        validate_rows(rows)
    metadata = json.loads((directory/"run_manifest.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    ids = metadata["partitions"]["eval_1000"]
    chosen = set(np.random.default_rng(config["seed"]).choice(
        sorted(ids), config["latency_images"], replace=False))
    devices = ("cpu", "cuda") if metadata["hardware"]["gpu"] else ("cpu",)
    expected = {(size, backend, device) for size in config["resolutions"]
                for backend in metadata["accuracy_backends"] for device in devices
                if backend != "onnx_fp16" or device == "cuda"}
    jobs = rows.groupby(["resolution", "backend", "device"])
    if set(jobs.groups) != expected:
        raise ValueError("Latency evidence is missing expected configuration/device pairs")
    for _, group in jobs:
        repeats = group.groupby("round")
        if set(repeats.groups) != set(range(1, config["latency_rounds"]+1)):
            raise ValueError("Latency evidence does not have the configured repetitions")
        for _, repeat in repeats:
            if len(repeat) != len(chosen) or set(repeat.image_id) != chosen:
                raise ValueError("Timing must use the same disjoint frozen image sample each round")
    if not rows.host_threads.eq(config["cpu_threads"]).all():
        raise ValueError("Latency evidence disagrees with the fixed thread configuration")
    return summarize_latency(rows)


def parity(frame):
    grid = frame.pivot(index="resolution", columns="backend", values="map")
    gaps = (grid.torch_fp32 - grid.onnx_fp32).abs()
    for size, gap in gaps.items():
        print(f"FP32 parity {size}: |delta mAP|={gap:.9f} (limit {PARITY_LIMIT:g})", flush=True)
    if gaps.empty or not gaps.le(PARITY_LIMIT).all():
        raise ValueError("PyTorch/ONNX FP32 parity failed")
    return {str(size): float(gap) for size, gap in gaps.items()}
