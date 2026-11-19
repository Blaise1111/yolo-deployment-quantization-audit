"""COCO area AP, COCO-compatible IoU-0.50 correctness and fixed-FPPI recall."""
from contextlib import redirect_stdout
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(ground_truth, records, limit, full=True):
    detections = []
    for image, record in records.items():
        for _, category, score, x1, y1, x2, y2 in record["detections"]:
            detections.append(dict(image_id=image, category_id=int(category), score=score,
                                   bbox=[x1, y1, x2-x1, y2-y1]))
    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {"info": {}, **ground_truth}
        truth.createIndex()
        if detections:
            result = truth.loadRes(detections)
        else:
            result = COCO()
            result.dataset = dict(images=ground_truth["images"], categories=ground_truth["categories"],
                                  annotations=[])
            result.createIndex()
        evaluation = COCOeval(truth, result, "bbox")
        evaluation.params.maxDets = [1, 10, 100, limit] if full else [limit]
        if not full:
            evaluation.params.iouThrs = np.array([0.5])
            evaluation.params.areaRng = evaluation.params.areaRng[:1]
            evaluation.params.areaRngLbl = ["all"]
        evaluation.evaluate()
        if full:
            evaluation.accumulate()
    metrics = {}
    if full:
        precision = evaluation.eval["precision"]
        standard = evaluation.params.maxDets.index(100)
        def average(grid):
            valid = grid[grid >= 0]
            return float(valid.mean()) if len(valid) else np.nan
        for index, name in enumerate(evaluation.params.areaRngLbl):
            key = "map" if name == "all" else "ap_" + name
            metrics[key] = average(precision[:, :, :, index, standard])
        metrics["ap50"] = average(precision[0, :, :, 0, standard])
        metrics["ap75"] = average(precision[5, :, :, 0, standard])

    scores, matched_ids, image_ids = [], [], []
    total = len(ground_truth["images"])
    for row in evaluation.evalImgs:
        if row is None or row["aRng"] != evaluation.params.areaRng[0]:
            continue
        valid = ~row["dtIgnore"][0]
        count = int(valid.sum())
        scores.extend(np.asarray(row["dtScores"])[valid])
        matched_ids.extend(row["dtMatches"][0, valid])
        image_ids.extend([row["image_id"]] * count)
    status = np.column_stack((scores, matched_ids, image_ids))
    return metrics, status, truth.anns, total


def fixed_fppi(status, gt, image_count, fppi):
    scores, matched, _ = status.T
    order = np.argsort(-scores, kind="stable")
    cutoffs = np.flatnonzero(np.r_[np.diff(scores[order]) != 0, True]) if len(order) else []
    false_counts = np.cumsum(matched[order] == 0)
    allowed = [i for i in cutoffs if false_counts[i] <= image_count * fppi]
    end = allowed[-1] + 1 if allowed else 0
    accepted = matched[order[:end]]
    captured = [gt[int(key)]["area"] for key in accepted if key]
    areas = [row["area"] for row in gt.values() if not row["iscrowd"]]
    out = dict(fppi_achieved=float((accepted == 0).sum() / image_count),
               confidence_at_fppi=float(scores[order[end-1]]) if end else np.nan)
    for name, low, high in (("small", 0, 32**2), ("medium", 32**2, 96**2),
                            ("large", 96**2, np.inf)):
        numerator = sum(low <= area < high for area in captured)
        denominator = sum(low <= area < high for area in areas)
        out[f"recall_{name}_fppi"] = numerator / denominator if denominator else np.nan
    return out
