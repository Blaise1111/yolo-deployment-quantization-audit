"""Export fused YOLO with a raw detection head, and calibrate static QDQ graphs."""
import json
import os
from pathlib import Path
import shutil
import tempfile

from .coco_split import ROOT, digest, write_json
from .unified_infer import decode_image, letterbox, model_path, torch_model


def export_fp32(size, threads):
    import onnx
    import torch

    path = model_path(size, "onnx_fp32")
    if path.is_file():
        return path
    torch.set_num_threads(threads)
    model = torch_model("cpu")
    example = torch.zeros(1, 3, size, size)
    temporary = path.with_suffix(".tmp")
    with torch.inference_mode():
        model(example)
        torch.onnx.export(model, example, temporary, input_names=["images"],
                          output_names=["predictions"], opset_version=17, dynamo=False)
    graph = onnx.load(str(temporary))
    onnx.checker.check_model(graph)
    if any(node.op_type == "NonMaxSuppression" for node in graph.graph.node):
        raise ValueError("Export must not include NMS")
    temporary.replace(path)
    return path


def export_variant(size, backend, calibration_ids, images, config):
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )

    base = export_fp32(size, config["cpu_threads"])
    path = model_path(size, backend)
    if backend == "onnx_fp32":
        return base
    record = dict(fp32_sha256=digest(base), precision=backend)
    if backend == "onnx_int8":
        record.update(calibration_ids=calibration_ids, method=config["calibration_method"])
    manifest = path.with_suffix(".provenance.json")
    if path.is_file() and manifest.is_file() and json.loads(manifest.read_text()) == record:
        return path
    temporary = path.with_suffix(".tmp")
    if backend == "onnx_fp16":
        from onnxconverter_common.float16 import convert_float_to_float16

        graph = convert_float_to_float16(onnx.load(str(base)), keep_io_types=True)
        onnx.save(graph, str(temporary))
    else:
        if len(calibration_ids) != 200 or len(set(calibration_ids)) != 200:
            raise ValueError("Static QDQ requires exactly 200 frozen calibration images")

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
                    per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                    calibrate_method=CalibrationMethod[config["calibration_method"]],
                    calibration_providers=["CPUExecutionProvider"],
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
    temporary.replace(path)
    write_json(manifest, record)
    print(f"Exported {path.name}", flush=True)
    return path
