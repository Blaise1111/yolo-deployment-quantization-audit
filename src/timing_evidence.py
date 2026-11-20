"""Timing contracts and digest-bound journals for new, explicitly measured runs."""
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .coco_split import IMAGES, digest, write_json

PHASES = ("decode", "preprocess", "inference", "postprocess")
KEY = ["resolution", "backend", "device", "round", "image_id"]


def path_for(path):
    return path.with_suffix(".timing.json")


def chosen_images(ids, config):
    return np.random.default_rng(config["seed"]).choice(
        sorted(ids), size=config["latency_images"], replace=False).tolist()


def contract(jobs, chosen, images, config, affinity):
    import onnxruntime as ort
    import torch
    from .latency_bench import hardware_metadata
    from .unified_infer import model_path, providers

    versions = {}
    for name in ("numpy","pandas","opencv-python","torch","torchvision","onnx","psutil","ultralytics"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError as exc:
            raise ValueError(f"Missing timing dependency {name}") from exc
    driver = None
    if any(device == "cuda" for _,_,device in jobs):
        try:
            driver=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version",
                                          "--format=csv,noheader"],text=True).strip()
        except (OSError,subprocess.CalledProcessError) as exc:
            raise ValueError("Cannot attest CUDA hardware and driver using nvidia-smi") from exc
    execution = []
    for size, backend, device in jobs:
        model = model_path(size,backend)
        if not model.is_file():
            raise FileNotFoundError(f"Missing benchmark model {model.name}")
        is_ort=backend.startswith("onnx")
        requested=providers(device) if is_ort else []
        execution.append(dict(size=size,backend=backend,device=device,model=model.name,
                              sha256=digest(model),providers=requested,
                              graph_optimization="ORT_ENABLE_ALL" if is_ort else None))
    result=dict(schema=1, units="ms", python=sys.version.split()[0], dependencies=versions,
                ort=ort.__version__, available_providers=ort.get_available_providers(),
                hardware=hardware_metadata(), cuda_driver=driver, torch_cuda=torch.version.cuda,
                torch_cudnn=torch.backends.cudnn.version(),
                affinity=list(affinity), intra_threads=config["cpu_threads"], inter_threads=1,
                opencv_threads=1, torch_tf32=False, warmup=config["warmups"],
                rounds=config["latency_rounds"], jobs=execution,
                sample_order=[dict(id=i, sha256=digest(IMAGES/images[i]["file_name"])) for i in chosen],
                preprocessing={k:config[k] for k in ("confidence","nms_iou","max_det")},
                implementation={filename:digest(Path(__file__).with_name(filename))
                                for filename in ("unified_infer.py","latency_bench.py","timing_evidence.py")})
    return json.loads(json.dumps(result))


def validate_rows(rows):
    required=KEY+[stage+"_ms" for stage in (*PHASES,"total")]
    if rows.empty or any(c not in rows for c in required):
        raise ValueError("Timing CSV is empty or missing sample keys or millisecond columns")
    if rows[KEY].isna().any().any() or rows.duplicated(KEY).any():
        raise ValueError("Missing or duplicate timing sample keys")
    values=rows[[s+"_ms" for s in (*PHASES,"total")]].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values<0).any()
            or not np.allclose(values[:,:-1].sum(axis=1),values[:,-1],atol=1e-7,rtol=0)):
        raise ValueError("Timing stages must be nonnegative finite milliseconds summing to elapsed total")


def validate_order(rows, expected, complete=True):
    validate_rows(rows)
    actual=list(rows[KEY].itertuples(index=False,name=None))
    ids=[r["id"] for r in expected["sample_order"]]
    planned=[(j["size"],j["backend"],j["device"],r,i) for j in expected["jobs"]
             for r in range(1,expected["rounds"]+1) for i in ids]
    per_job=len(ids)*expected["rounds"]
    if len(actual)%per_job or actual!=planned[:len(actual)] or (complete and len(actual)!=len(planned)):
        raise ValueError("Timing sample/job order or coverage differs from the manifest")
    affinity=",".join(map(str,expected["affinity"]))
    if not (rows.host_threads.eq(expected["intra_threads"]).all() and
            rows.host_affinity.astype(str).eq(affinity).all()):
        raise ValueError("Timing row affinity or thread count differs from the manifest")


def publish(path, expected, complete=True, origin="new_measurement", actual_provider_options=None):
    write_json(path_for(path),dict(origin=origin,complete=complete,
               sha256=digest(path),contract=expected,
               actual_provider_options=actual_provider_options or {}))


def load(path, expected=None, complete=True):
    marker=path_for(path)
    if not path.is_file() or not marker.is_file():
        raise ValueError(f"Missing timing CSV or attested manifest for {path.name}; legacy timing is offline-only")
    info=json.loads(marker.read_text(encoding="utf-8"))
    if info.get("sha256")!=digest(path) or info.get("complete") is not complete:
        raise ValueError("Timing file digest or completion marker differs from manifest")
    if expected is not None and expected!=info["contract"]:
        changes=sorted(key for key in expected.keys()|info["contract"].keys()
                       if expected.get(key)!=info["contract"].get(key))
        raise ValueError("Incompatible timing manifest: " + ", ".join(changes))
    for trace in info.get("actual_provider_options", {}).values():
        filename = Path(trace["trace"])
        if filename.name != str(filename) or not (path.parent/filename).is_file():
            raise ValueError(f"Missing or invalid timing provider trace {filename.name}")
        if digest(path.parent/filename) != trace["trace_sha256"]:
            raise ValueError(f"Provider trace digest mismatch: {filename.name}")
    rows=pd.read_csv(path,float_precision="round_trip",dtype={"host_affinity":str})
    validate_order(rows,info["contract"],complete)
    return rows,info


def reuse(source, target, expected):
    _,info=load(source,expected)
    temporary=target.with_suffix(".reuse")
    shutil.copyfile(source,temporary)
    if digest(temporary)!=info["sha256"]:
        temporary.unlink()
        raise ValueError("Timing source changed during reuse")
    for trace in info["actual_provider_options"].values():
        name = trace["trace"]
        if source.parent.resolve() != target.parent.resolve():
            shutil.copyfile(source.parent/name, target.parent/name)
        if digest(target.parent/name) != trace["trace_sha256"]:
            raise ValueError(f"Provider trace changed during reuse: {name}")
    temporary.replace(target)
    write_json(path_for(target),{**info,"origin":"reused_verified_measurement"})
