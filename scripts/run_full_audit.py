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


def render_reports(evidence, output):
    from src.reports import REPORTS, csv_write, detection_reports, latency_report, parity
    from src.plotting import create_plots

    try:
        reports, details = detection_reports(evidence)
        reports[REPORTS[3]] = latency_report(evidence)
    except FileNotFoundError as exc:
        raise ValueError(f"Saved evidence is incomplete: {exc.filename}. Reports-only never runs inference.") from exc
    gaps = parity(reports[REPORTS[1]])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        csv_write(frame, output/f"{name}.csv")
    create_plots(reports, details, output/"figures")
    return gaps


def initialize_evidence(path, data):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError(f"Incompatible existing evidence: {path.name}; select a new output directory")
    else:
        write_json(path, data)


def run(config, directory, mode, fast_gate_dir, predictions_only=False, reuse_timing_from=None):
    import psutil
    import torch
    from src.detections import save_records
    from src.export_quantize import export_variant
    from src.parity_gate import GATE_NAME, parity_contract, publish_gate, read_gate, validate_gate
    from src.latency_bench import benchmark, hardware_metadata
    from src.reports import detection_file
    from src.timing_evidence import chosen_images, contract, load as load_timing, path_for, reuse
    from src.unified_infer import model_path, providers
    import onnxruntime as ort

    fast = mode == "eval_1000"
    gate_record = None if fast else read_gate(fast_gate_dir / GATE_NAME)
    torch.set_num_interop_threads(1)
    source, images, partitions = coco_inputs(config)
    categories = sorted(source["categories"], key=lambda row: row["id"])
    calib, ids = partitions["calib_200"], partitions[mode]
    process = psutil.Process()
    affinity = process.cpu_affinity()

    def export(size, backend):
        process.cpu_affinity(affinity[:config["cpu_threads"]])
        try:
            export_variant(size, backend, calib, images, config)
        finally:
            process.cpu_affinity(affinity)

    if fast:
        for size in config["resolutions"]:
            export(size, "onnx_fp32")
    gate_contract = parity_contract(config, partitions, images)
    if gate_record is not None:
        validate_gate(gate_record, gate_contract)
    cuda = torch.cuda.is_available()
    preferred = "cuda" if cuda else "cpu"
    backends = dict(torch_fp32=preferred, onnx_fp32=preferred, onnx_int8="cpu")
    if cuda and config["fp16_cuda"]:
        backends["onnx_fp16"] = "cuda"
    model_runtime = dict(torch=torch.__version__, ort=ort.__version__, cuda=torch.version.cuda,
                         graph_optimization="ORT_ENABLE_ALL", requested_providers={
                             device: providers(device) for device in set(backends.values())})
    model_runtime=json.loads(json.dumps(model_runtime))
    metadata = dict(config=config, partitions=partitions, eval_partition=mode,
                    accuracy_backends=backends, annotations_sha256=digest(ANNOTATIONS),
                    hardware=hardware_metadata(), prediction_runtime=model_runtime)
    initialize_evidence(directory/"run_manifest.json", metadata)
    initialize_evidence(directory/"eval_ground_truth.json", ground_truth(source, ids))
    initialize_evidence(directory/"calib_ground_truth.json", ground_truth(source, calib))
    for size in config["resolutions"]:
        for backend, device in backends.items():
            if backend != "torch_fp32":
                export(size, backend)
            model = model_path(size, backend)
            model_sha256 = digest(model)
            expected = {"torch_fp32": gate_contract["checkpoint_sha256"],
                        "onnx_fp32": gate_contract["fp32_sha256"][str(size)]}.get(backend)
            if expected is not None and model_sha256 != expected:
                raise ValueError(f"{model.name} changed after parity preflight; restart with stable inputs")
            provenance = dict(config=config, ids=ids, device=device, model_sha256=model_sha256,
                              annotations_sha256=metadata["annotations_sha256"], runtime=model_runtime)
            save_records(detection_file(directory, size, backend), size, backend, device,
                         ids, images, categories, config, provenance)
            if backend == "onnx_int8":
                save_records(directory/f"calib_int8_{size}.jsonl", size, backend, device,
                             calib, images, categories, config, {**provenance, "ids": calib})
    if predictions_only:
        return
    timings = directory/"latency_logs.csv"
    jobs = [(size, backend, device) for size in config["resolutions"]
            for backend in backends for device in ("cpu", "cuda")
            if (device != "cuda" or cuda) and (backend != "onnx_fp16" or device == "cuda")]
    if timings.exists() or path_for(timings).exists() or reuse_timing_from is not None:
        expected = contract(jobs, chosen_images(partitions["eval_1000"],config), images, config,
                            affinity[:config["cpu_threads"]])
        if timings.exists() or path_for(timings).exists():
            load_timing(timings, expected)
        else:
            reuse(reuse_timing_from/"latency_logs.csv", timings, expected)
    else:
        benchmark(timings, jobs, images, partitions["eval_1000"], categories, config)
    gaps = render_reports(directory,directory)
    if fast:
        if gate_contract != parity_contract(config, partitions, images):
            raise ValueError("Fast parity inputs changed during evaluation; rerun with stable inputs")
        publish_gate(directory / GATE_NAME, gate_contract, gaps)
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
    parser.add_argument("--results-dir", type=Path, help="Read-only prediction and timing evidence")
    parser.add_argument("--output-dir", type=Path, help="Dedicated new run or report directory")
    parser.add_argument("--reuse-timing-from", type=Path, help="Reuse only a compatible attested timing capture")
    parser.add_argument("--fast-gate-dir", type=Path, default=ROOT/"audit_outputs/new_fast")
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
    if args.output_dir is None:
        parser.error("New runs and reports require a separate --output-dir")
    output = args.output_dir.resolve()
    historical = (ROOT/"results").resolve()
    if output == historical or historical in output.parents:
        parser.error("--output-dir must be outside historical results")
    if args.reports_only:
        evidence=(args.results_dir or directory).resolve()
        if output == evidence or evidence in output.parents:
            parser.error("--reports-only requires separate input and output directories")
        render_reports(evidence,output)
        return
    if args.results_dir is not None:
        parser.error("--results-dir is only for offline reports or verification")
    (ROOT/"logs").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT/"logs"))
    (ROOT/"models").mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    from src.export_quantize import ExportCacheError
    from src.parity_gate import ParityGateError

    try:
        run(config, output, "eval_1000" if fast else "eval_4800", args.fast_gate_dir,
            args.predictions_only, args.reuse_timing_from)
    except (ExportCacheError, ParityGateError) as exc:
        parser.error(str(exc))



if __name__ == "__main__":
    main()
