"""Measure serial decode, Letterbox, synchronized forward and external NMS."""
from collections import Counter
import platform
from pathlib import Path
from time import perf_counter_ns
import json

import numpy as np
import pandas as pd
import psutil

from .unified_infer import Runner, decode_image, letterbox, postprocess, model_path

STAGES = ("decode", "preprocess", "inference", "postprocess", "total")


def hardware_metadata():
    import torch

    return dict(cpu=platform.processor(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "",
                torch=torch.__version__, platform=platform.platform())


def benchmark(path, jobs, images, ids, categories, config):
    import torch

    process = psutil.Process()
    mask = process.cpu_affinity()
    intended = mask[:config["cpu_threads"]]
    if len(intended) != config["cpu_threads"]:
        raise ValueError("The CPU benchmark requires four permitted logical CPUs")
    rng = np.random.default_rng(config["seed"])
    chosen = rng.choice(sorted(ids), size=config["latency_images"], replace=False).tolist()
    category_ids = [item["id"] for item in categories]

    def one(runner, image_id, resolution):
        start = perf_counter_ns()
        image = decode_image(images[image_id]["file_name"])
        decoded = perf_counter_ns()
        tensor, geometry = letterbox(image, resolution)
        preprocessed = perf_counter_ns()
        output = runner(tensor)  # Returns CPU arrays after CUDA work completes.
        inferred = perf_counter_ns()
        postprocess(output, geometry, category_ids, config)
        finished = perf_counter_ns()
        ticks = (decoded-start, preprocessed-decoded, inferred-preprocessed,
                 finished-inferred, finished-start)
        return {name + "_ms": ns / 1e6 for name, ns in zip(STAGES, ticks)}

    partial = path.with_name("latency_partial.csv")
    rows = pd.read_csv(partial, float_precision="round_trip").to_dict("records") if partial.is_file() else []
    done = {(row["resolution"], row["backend"], row["device"]) for row in rows}
    try:
        process.cpu_affinity(intended)
        for resolution, backend, device in jobs:
            if (resolution, backend, device) in done:
                continue
            is_onnx = backend.startswith("onnx")
            runner = Runner(resolution, backend, device, config["cpu_threads"], profile=is_onnx)
            try:
                placement = {}
                int8_placement = {}
                if is_onnx:
                    one(runner, chosen[0], resolution)
                    trace = json.loads(Path(runner.runtime.end_profiling()).read_text(encoding="utf-8"))
                    nodes = [event["args"] for event in trace if event.get("cat") == "Node"
                             and "provider" in event.get("args", {})]
                    placement = dict(Counter(node["provider"] for node in nodes))
                    quantized_ops = {"QLinearConv", "QLinearMatMul", "QGemm", "MatMulInteger",
                                     "ConvInteger", "QLinearConvTranspose"}
                    int8_placement = dict(Counter(node["provider"] for node in nodes
                                                  if node.get("op_name") in quantized_ops))
                for i in range(config["warmups"]):
                    one(runner, chosen[i % len(chosen)], resolution)
                if device == "cuda" and backend == "torch_fp32":
                    torch.cuda.reset_peak_memory_stats()
                weight_bytes = model_path(resolution, backend).stat().st_size
                for repeat in range(config["latency_rounds"]):
                    for image_id in chosen:
                        measured = one(runner, image_id, resolution)
                        vram = (torch.cuda.max_memory_allocated() if
                                device == "cuda" and backend == "torch_fp32" else np.nan)
                        rows.append(dict(resolution=resolution, backend=backend, device=device,
                                         round=repeat+1, image_id=image_id, **measured,
                                         rss_bytes=process.memory_info().rss,
                                         model_bytes=weight_bytes, torch_peak_cuda_bytes=vram,
                                         host_threads=config["cpu_threads"],
                                         host_affinity=",".join(map(str, intended)),
                                         node_placement=json.dumps(placement, sort_keys=True),
                                         int8_compute_placement=json.dumps(int8_placement, sort_keys=True)))
            finally:
                runner.close()
            checkpoint = partial.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(checkpoint, index=False, float_format="%.17g")
            checkpoint.replace(partial)
            print(f"Latency measured: {resolution} {backend} {device}", flush=True)
    finally:
        process.cpu_affinity(mask)
    temporary = path.with_suffix(".part")
    pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)
    partial.unlink(missing_ok=True)


def summarize_latency(rows):
    summary = []
    for (size, backend, device), run in rows.groupby(["resolution", "backend", "device"]):
        result = dict(resolution=size, backend=backend, device=device, samples=len(run),
                      relative_area=(size/640)**2, model_mib=run.model_bytes.iloc[0]/2**20,
                      peak_sampled_rss_mib=run.rss_bytes.max()/2**20,
                      torch_peak_cuda_mib=run.torch_peak_cuda_bytes.max()/2**20,
                      host_threads=int(run.host_threads.iloc[0]),
                      host_affinity=run.host_affinity.iloc[0],
                      node_placement=run.node_placement.iloc[0],
                      int8_compute_placement=run.int8_compute_placement.iloc[0])
        for stage in STAGES:
            samples = run[stage + "_ms"]
            result.update({f"{stage}_mean_ms": samples.mean(), f"{stage}_p50_ms": samples.median(),
                           f"{stage}_p95_ms": samples.quantile(.95)})
        summary.append(result)
    table = pd.DataFrame(summary)
    for (backend, device), batch in table.groupby(["backend", "device"]):
        anchor = batch.loc[batch.resolution.eq(640), "total_p95_ms"].iloc[0]
        table.loc[batch.index, "total_p95_speedup_vs_640"] = anchor / batch.total_p95_ms
    return table
