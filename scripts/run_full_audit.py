"""Run a frozen fast or full COCO audit, or verify saved summaries in isolation."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coco_split import ANNOTATIONS, ROOT, coco_inputs, digest, ground_truth, load_config, write_json


def verify(directory):
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from src.reports import REPORTS, csv_write, detection_reports, latency_report, parity

    reports, _ = detection_reports(directory)
    reports[REPORTS[3]] = latency_report(directory)
    parity(reports[REPORTS[1]])
    with TemporaryDirectory() as scratch:
        for name, table in reports.items():
            temporary = Path(scratch)/f"{name}.csv"
            csv_write(table, temporary)
            original = pd.read_csv(directory/temporary.name, float_precision="round_trip")
            regenerated = pd.read_csv(temporary, float_precision="round_trip")
            assert_frame_equal(original, regenerated, check_exact=False, atol=1e-7, rtol=0)
            print(f"PASS {temporary.name} at absolute tolerance 1e-7", flush=True)


def run(config, directory, mode, predictions_only=False, reports_only=False):
    import psutil
    import torch
    from src.detections import save_records
    from src.export_quantize import export_variant
    from src.latency_bench import benchmark, hardware_metadata
    from src.reports import REPORTS, csv_write, detection_file, detection_reports, latency_report, parity
    from src.unified_infer import model_path

    torch.set_num_interop_threads(1)
    source, images, partitions = coco_inputs(config)
    categories = sorted(source["categories"], key=lambda row: row["id"])
    calib, ids = partitions["calib_200"], partitions[mode]
    cuda = torch.cuda.is_available()
    preferred = "cuda" if cuda else "cpu"
    backends = dict(torch_fp32=preferred, onnx_fp32=preferred, onnx_int8="cpu")
    if cuda and config["fp16_cuda"]:
        backends["onnx_fp16"] = "cuda"
    metadata = dict(config=config, partitions=partitions, eval_partition=mode,
                    accuracy_backends=backends, annotations_sha256=digest(ANNOTATIONS),
                    hardware=hardware_metadata())
    write_json(directory/"run_manifest.json", metadata)
    write_json(directory/"eval_ground_truth.json", ground_truth(source, ids))
    write_json(directory/"calib_ground_truth.json", ground_truth(source, calib))
    if not reports_only:
        for size in config["resolutions"]:
            for backend, device in backends.items():
                if backend != "torch_fp32":
                    process = psutil.Process()
                    available = process.cpu_affinity()
                    try:
                        process.cpu_affinity(available[:config["cpu_threads"]])
                        export_variant(size, backend, calib, images, config)
                    finally:
                        process.cpu_affinity(available)
                model = model_path(size, backend)
                provenance = dict(config=config, ids=ids, device=device, model_sha256=digest(model),
                                  annotations_sha256=metadata["annotations_sha256"])
                save_records(detection_file(directory, size, backend), size, backend, device,
                             ids, images, categories, config, provenance)
                if backend == "onnx_int8":
                    save_records(directory/f"calib_int8_{size}.jsonl", size, backend, device,
                                 calib, images, categories, config, {**provenance, "ids": calib})
    if predictions_only:
        return
    reports, details = detection_reports(directory)
    parity(reports[REPORTS[1]])
    timings = directory/"latency_logs.csv"
    if not timings.is_file():
        jobs = [(size, backend, device) for size in config["resolutions"]
                for backend in backends for device in ("cpu", "cuda")
                if (device != "cuda" or cuda) and (backend != "onnx_fp16" or device == "cuda")]
        # Use the same benchmark images in both modes; reuse the measurements on the same host.
        fast_timings = ROOT/"results"/"fast_eval_1000"/timings.name
        fast_manifest = fast_timings.parent/"run_manifest.json"
        compatible_timing = (mode == "eval_4800" and fast_timings.is_file()
                             and fast_manifest.is_file())
        if compatible_timing:
            previous = json.loads(fast_manifest.read_text(encoding="utf-8"))
            compatible_timing = all(previous[key] == metadata[key] for key in (
                "config", "hardware", "accuracy_backends", "annotations_sha256"))
        if compatible_timing:
            from shutil import copyfile
            copyfile(fast_timings, timings)
        else:
            benchmark(timings, jobs, images, partitions["eval_1000"], categories, config)
    reports[REPORTS[3]] = latency_report(directory)
    for name, frames in reports.items():
        csv_write(frames, directory/f"{name}.csv")
    from src.plotting import create_plots
    output = ROOT/"figs"/"fast_eval_1000" if mode == "eval_1000" else ROOT/"figs"
    create_plots(reports, details, output)
    if mode == "eval_1000":
        write_json(directory/"fp32_parity_pass.json", dict(config=config,
                  annotations_sha256=metadata["annotations_sha256"]))
    print(f"Finished {mode}: {len(ids)} images", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--fast-eval-1000", action="store_true")
    scope.add_argument("--full", action="store_true")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--verify-csv", action="store_true")
    operation.add_argument("--predictions-only", action="store_true")
    operation.add_argument("--reports-only", action="store_true")
    parser.add_argument("--results-dir", type=Path, help="Existing evidence directory for --verify-csv")
    args = parser.parse_args()
    if args.verify_csv and args.results_dir:
        verify(args.results_dir)
        return
    config = load_config()
    configured_fast = config.pop("fast_eval_1000")
    fast = (args.fast_eval_1000 or configured_fast) and not args.full
    directory = ROOT/"results"/"fast_eval_1000" if fast else ROOT/"results"
    if args.verify_csv:
        verify(directory)
        return
    (ROOT/"logs").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"logs"))
    (ROOT/"models").mkdir(exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    if not fast:
        gate = ROOT/"results"/"fast_eval_1000"/"fp32_parity_pass.json"
        if (not (gate.parent/"p2_latency_scaling_breakdown.csv").is_file()
                or not gate.is_file() or json.loads(gate.read_text()) != dict(
                config=config, annotations_sha256=digest(ANNOTATIONS))):
            parser.error("Run the fast evaluation and pass FP32 parity before full inference")
    run(config, directory, "eval_1000" if fast else "eval_4800", args.predictions_only,
        args.reports_only)


if __name__ == "__main__":
    main()
