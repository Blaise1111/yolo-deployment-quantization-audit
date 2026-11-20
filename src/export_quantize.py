"""Export fused YOLO with a raw detection head, and calibrate static QDQ graphs."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile

from .coco_split import IMAGES, WEIGHTS, digest, write_json
from .unified_infer import decode_image, letterbox, model_path, torch_model


FP32_OPTIONS = dict(input_names=["images"], output_names=["predictions"], opset_version=17, dynamo=False)
FP16_OPTIONS = dict(keep_io_types=True)
INT8_OPTIONS = dict(per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                    calibration_providers=["CPUExecutionProvider"])


class ExportCacheError(ValueError):
    pass


def image_fingerprints(ids, images):
    """Ordered image IDs and file SHA-256, following the timing evidence contract."""
    return [dict(id=i, sha256=digest(IMAGES / images[i]["file_name"])) for i in ids]


def fp32_contract(size, threads):
    return dict(schema=1, backend="onnx_fp32", checkpoint_sha256=digest(WEIGHTS),
                input_shape=[1, 3, size, size], threads=threads, export=FP32_OPTIONS,
                dependencies={name: version(name) for name in ("torch", "ultralytics", "onnx")},
                implementation={name: digest(Path(__file__).with_name(name))
                                for name in ("export_quantize.py", "unified_infer.py")})


def cache_fingerprint(path, expected):
    """Return the validated artifact digest, or None for a completely empty cache slot."""
    manifest = path.with_suffix(".provenance.json")
    if not path.exists() and not manifest.exists():
        return None
    hint = f"Move {path.name} and {manifest.name} out of models/, then rerun --fast-eval-1000 in a fresh output directory."
    if not path.is_file() or not manifest.is_file():
        raise ExportCacheError(f"Incomplete export cache for {path.name}. {hint}")
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ExportCacheError(f"Unreadable export cache manifest for {path.name}. {hint}") from exc
    if not isinstance(record, dict) or record.get("contract") != expected:
        raise ExportCacheError(f"Export cache inputs or configuration differ for {path.name}. {hint}")
    checksum = digest(path)
    if record.get("artifact_sha256") != checksum:
        raise ExportCacheError(f"Export cache artifact SHA-256 differs for {path.name}. {hint}")
    return checksum


def publish_export(path, temporary, contract):
    checksum = digest(temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".provenance.json"), dict(contract=contract, artifact_sha256=checksum))


def validated_fp32_sha256(size, threads):
    path = model_path(size, "onnx_fp32")
    checksum = cache_fingerprint(path, fp32_contract(size, threads))
    if checksum is None:
        raise ExportCacheError(f"Missing attested export {path.name}. Run --fast-eval-1000 with a fresh output directory.")
    return checksum


def export_fp32(size, threads):
    path = model_path(size, "onnx_fp32")
    expected = fp32_contract(size, threads)
    if cache_fingerprint(path, expected) is not None:
        return path

    import onnx
    import torch
    torch.set_num_threads(threads)
    model = torch_model("cpu")
    example = torch.zeros(1, 3, size, size)
    temporary = path.with_suffix(".tmp")
    with torch.inference_mode():
        model(example)
        torch.onnx.export(model, example, temporary, **FP32_OPTIONS)
    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph)
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise ValueError("Export must not include NMS")
    if expected != fp32_contract(size, threads):
        raise ExportCacheError("FP32 export inputs changed during export. Restore the intended checkpoint and retry.")
    publish_export(path, temporary, expected)
    return path


def variant_contract(base, size, backend, calibration_ids, images, config):
    record = dict(schema=1, backend=backend, resolution=size, fp32_sha256=digest(base),
                  implementation=digest(Path(__file__)))
    if backend == "onnx_fp16":
        return {**record, "conversion": FP16_OPTIONS,
                "dependencies": {name: version(name) for name in ("onnx", "onnxconverter-common")}}
    if backend != "onnx_int8":
        raise ValueError(f"Unsupported conversion backend {backend}")
    if len(calibration_ids) != 200 or len(set(calibration_ids)) != 200:
        raise ValueError("Static QDQ requires exactly 200 frozen calibration images")
    import onnxruntime as ort

    return {**record, "conversion": dict(INT8_OPTIONS, format="QDQ", activation="QInt8", weight="QInt8",
                                         method=config["calibration_method"]),
            "calibration_images": image_fingerprints(calibration_ids, images),
            "preprocessing_sha256": digest(Path(__file__).with_name("unified_infer.py")),
            "dependencies": dict(onnx=version("onnx"), ort=ort.__version__,
                                 opencv=version("opencv-python"), numpy=version("numpy"))}


def export_variant(size, backend, calibration_ids, images, config):
    base = export_fp32(size, config["cpu_threads"])
    path = model_path(size, backend)
    if backend == "onnx_fp32":
        return base
    record = variant_contract(base, size, backend, calibration_ids, images, config)
    if cache_fingerprint(path, record) is not None:
        return path

    import onnx

    temporary = path.with_suffix(".tmp")
    if backend == "onnx_fp16":
        from onnxconverter_common.float16 import convert_float_to_float16

        graph = convert_float_to_float16(onnx.load(str(base)), **FP16_OPTIONS)
        onnx.save(graph, str(temporary))
    else:
        from onnxruntime.quantization import (
            CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
        )

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.ids = iter(calibration_ids)

            def get_next(self):
                image_id = next(self.ids, None)
                if image_id is None:
                    return None
                pixels = decode_image(images[image_id]["file_name"])
                return {"images": letterbox(pixels, size)[0]}

            def rewind(self):
                self.ids = iter(calibration_ids)

        # ORT's path-based ONNX shape inference uses a narrow Windows path.
        # Stage only temporary files on that platform when the checkout or TEMP is Unicode.
        default_temp = tempfile.gettempdir()
        stage = None
        if os.name == "nt" and not (str(base).isascii() and default_temp.isascii()):
            stage = Path(os.environ["PUBLIC"]) / "Documents"
        with tempfile.TemporaryDirectory(prefix="p2_onnx_", dir=stage) as work:
            scratch = Path(work)
            original = scratch / "fp32.onnx"
            quantized = scratch / "int8.onnx"
            shutil.copyfile(base, original)
            previous_temp = tempfile.tempdir
            try:
                tempfile.tempdir = str(scratch)
                quantize_static(
                    str(original), str(quantized), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod[config["calibration_method"]],
                    **INT8_OPTIONS,
                )
            finally:
                tempfile.tempdir = previous_temp
            shutil.copyfile(quantized, temporary)
    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph)
    operations = {node.op_type for node in graph.graph.node}
    if "NonMaxSuppression" in operations:
        raise ValueError("Graph contains internal NMS")
    if backend == "onnx_int8" and not {"QuantizeLinear", "DequantizeLinear"} <= operations:
        raise ValueError("Graph does not contain static QDQ operators")
    if record != variant_contract(base, size, backend, calibration_ids, images, config):
        raise ExportCacheError(f"Conversion inputs changed during export of {path.name}. Retry with stable inputs.")
    publish_export(path, temporary, record)
    print(f"Exported {path.name}", flush=True)
    return path
