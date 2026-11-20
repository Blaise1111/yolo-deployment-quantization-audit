"""Shared image transforms, dense candidate decoding and class-aware NMS."""
from dataclasses import dataclass
import gc

import cv2
import numpy as np

from .coco_split import IMAGES, WEIGHTS


@dataclass(frozen=True)
class Geometry:
    height: int
    width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - w) // 2, (size - h) // 2
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top+h, left:left+w] = resized
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor *= np.float32(1 / 255)
    return tensor, Geometry(height, width, h, w, top, left)


def restore_boxes(boxes, geometry):
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry.left) * geometry.width / geometry.resized_width
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry.top) * geometry.height / geometry.resized_height
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, geometry.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, geometry.height)
    return boxes


def decode_image(filename):
    # imdecode supports Windows paths outside the active code page.
    data = np.fromfile(IMAGES / filename, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode COCO image {filename}")
    return image


def postprocess(output, geometry, categories, config, record_candidates=False):
    import torch
    from torchvision.ops import nms

    raw = np.asarray(output[0]).T
    if raw.shape[1] != 4 + len(categories):
        raise ValueError(f"Unexpected detector output shape {raw.shape}")
    scores = raw[:, 4:].max(axis=1)
    ids = np.flatnonzero(scores >= config["confidence"])
    group = raw[ids]
    labels = group[:, 4:].argmax(axis=1)
    class_ids = np.asarray(categories)[labels]
    centers, halves = group[:, :2], group[:, 2:4] * 0.5
    boxes = np.column_stack((centers - halves, centers + halves)).astype(np.float32, copy=False)
    # Sort by anchor for tied scores, shared by every backend. Offset classes before IoU.
    order = np.argsort(-scores[ids], kind="stable")
    sorted_boxes = boxes[order]
    span = float(np.ptp(boxes)) + 1 if len(boxes) else 1
    translated = sorted_boxes + (labels[order] * span).astype(np.float32)[:, None]
    # Use ranks as unique NMS scores. Equal model scores retain canonical anchor order.
    ranking = torch.arange(len(order), 0, -1, dtype=torch.float32)
    kept = order[nms(torch.from_numpy(translated), ranking, config["nms_iou"]).numpy()]
    kept = kept[:config["max_det"]]
    restored = restore_boxes(boxes[kept], geometry)
    valid = (restored[:, 2] > restored[:, 0]) & (restored[:, 3] > restored[:, 1])
    def rows(index, coordinates):
        return np.column_stack((ids[index], class_ids[index], scores[ids[index]], coordinates)).tolist()
    records = {"detections": rows(kept[valid], restored[valid])}
    if record_candidates:
        audit = np.flatnonzero(scores[ids] >= config["operating_confidence"] - 0.05)
        records["candidates"] = rows(audit, boxes[audit])
    return records


def torch_model(device):
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS)).model.eval().fuse(verbose=False).to(device)
    head = model.model[-1]
    head.export = True
    head.format = "onnx"
    head.dynamic = False
    return model


def providers(device):
    if device == "cuda":
        return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def ort_session(path, device, threads, profile=False):
    import onnxruntime as ort
    if device == "cuda":
        import torch  # load the CUDA and cuDNN DLLs provided by the torch wheel
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if profile:
        from .coco_split import ROOT
        options.enable_profiling = True
        options.profile_file_prefix = str(ROOT / "logs" / f"{path.stem}_placement")
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers(device))
    requested = providers(device)[0]
    requested = requested[0] if isinstance(requested, tuple) else requested
    if session.get_providers()[0] != requested:
        raise RuntimeError(f"ORT failed to initialize {requested}")
    session.disable_fallback()
    return session


def model_path(resolution, backend):
    from .coco_split import ROOT
    return WEIGHTS if backend == "torch_fp32" else ROOT / "models" / f"{backend}_{resolution}.onnx"


class Runner:
    def __init__(self, resolution, backend, device, threads, profile=False):
        import torch
        torch.set_num_threads(threads)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        cv2.setNumThreads(1)
        self.backend, self.device = backend, device
        self.path = model_path(resolution, backend)
        self.runtime = torch_model(device) if backend == "torch_fp32" else ort_session(
            self.path, device, threads, profile)
        self.names = self.runtime.names if backend == "torch_fp32" else None

    def __call__(self, tensor):
        if self.backend == "torch_fp32":
            import torch
            with torch.inference_mode():
                return self.runtime(torch.from_numpy(tensor).to(self.device)).cpu().numpy()
        return self.runtime.run(None, {"images": tensor})[0]

    def close(self):
        self.runtime = None
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()
