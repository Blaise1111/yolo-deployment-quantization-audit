"""Paired box drift, evidence-based flip attribution, and calibration metrics."""
from collections import Counter

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def iou(boxes, other):
    extent = np.maximum(np.minimum(boxes[:, None, 2:], other[None, :, 2:])
                        - np.maximum(boxes[:, None, :2], other[None, :, :2]), 0)
    overlap = extent.prod(axis=2)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    other_areas = (other[:, 2:] - other[:, :2]).prod(axis=1)
    union = areas[:, None] + other_areas[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def greedy_matches(first, second, threshold):
    if not len(first) or not len(second):
        return []
    overlaps = iou(first[:, 3:], second[:, 3:])
    same = first[:, None, 1] == second[None, :, 1]
    i, j = np.where(same & (overlaps >= threshold))
    order = np.argsort(-overlaps[i, j], kind="stable")
    used_i, used_j, out = set(), set(), []
    for index in order:
        a, b = i[index], j[index]
        if a not in used_i and b not in used_j:
            out.append((a, b, overlaps[a, b]))
            used_i.add(a)
            used_j.add(b)
    return out


def flip_rate(matched, lost, added):
    union = matched + lost + added
    return (lost + added) / union if union else 0.0


def suppressed_by(candidate, records, kept, nms_iou):
    peers = [r for key, r in records.items()
             if key in kept and r[1] == candidate[1] and r[0] != candidate[0]
             and (r[2], -r[0]) > (candidate[2], -candidate[0])]
    if not peers:
        return set()
    boxes = np.asarray(peers)
    overlaps = iou(candidate[None, 3:], boxes[:, 3:])[0]
    return set(boxes[overlaps > nms_iou, 0])


def attribution(row, present, absent, present_kept, absent_kept, config):
    anchor, category, score = row[:3]
    cut = config["operating_confidence"]
    alternate = absent.get(anchor)
    if (cut - 0.05 <= score <= cut + 0.05
            and (alternate is None or (alternate[1] == category and alternate[2] < cut))):
        return "threshold_edge"
    original = present.get(anchor)
    if (original is not None and alternate is not None and alternate[1] == category
            and alternate[2] >= cut and anchor not in absent_kept):
        opponents = suppressed_by(alternate, absent, absent_kept, config["nms_iou"])
        for key in opponents - present_kept:
            former = present.get(key)
            if former is not None and former[2] >= cut and former[1] == category:
                if anchor in suppressed_by(former, present, present_kept, config["nms_iou"]):
                    return "nms_reversal"
    return "unresolved"


def as_boxes(rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def audit_pair(reference, variant, config):
    cut = config["operating_confidence"]
    matched, lost, added = 0, 0, 0
    residuals, flip_scores, overlaps = [], [], []
    counts = Counter()
    for image, record in reference.items():
        other = variant[image]
        a, b = (as_boxes(r["detections"]) for r in (record, other))
        a, b = a[a[:, 2] >= cut], b[b[:, 2] >= cut]
        if not len(a) and not len(b):
            continue
        pairs = greedy_matches(a, b, config["match_iou"])
        matched += len(pairs)
        overlaps.extend(overlap for _, _, overlap in pairs)
        residuals.extend((a[i, 2], b[j, 2] - a[i, 2]) for i, j, _ in pairs)
        seen_a = {i for i, _, _ in pairs}
        seen_b = {j for _, j, _ in pairs}
        lost += len(a) - len(seen_a)
        added += len(b) - len(seen_b)
        ca, cb = ({r[0]: r for r in as_boxes(x["candidates"])} for x in (record, other))
        kept_a, kept_b = set(a[:, 0]), set(b[:, 0])
        for rows, seen, first, second, kept_first, kept_second, action in (
            (a, seen_a, ca, cb, kept_a, kept_b, "lost"),
            (b, seen_b, cb, ca, kept_b, kept_a, "added"),
        ):
            for index, row in enumerate(rows):
                if index not in seen:
                    reason = attribution(row, first, second, kept_first, kept_second, config)
                    counts[reason] += 1
                    flip_scores.append((row[2], action, reason))
    absolute = np.abs([delta for _, delta in residuals])
    total = lost + added
    result = dict(matched=matched, lost=lost, added=added, flip_rate=flip_rate(matched, lost, added),
                  mean_iou=np.mean(overlaps) if matched else np.nan,
                  abs_delta_p50=np.quantile(absolute, 0.5) if matched else np.nan,
                  abs_delta_p95=np.quantile(absolute, 0.95) if matched else np.nan)
    for reason in ("threshold_edge", "nms_reversal", "unresolved"):
        result[reason + "_count"] = counts[reason]
        result[reason + "_share"] = counts[reason] / total if total else 0.0
    return result, residuals, flip_scores


def transform(scores, temperature):
    return expit(logit(np.asarray(scores, dtype=np.float64)) / temperature)


def temperature_fit(scores, truth):
    if not len(scores) or len(np.unique(truth)) != 2:
        raise ValueError("Confidence calibration requires both correct and incorrect detections")
    logits = logit(np.clip(scores, 1e-12, 1 - 1e-12))
    best = minimize_scalar(lambda t: np.mean(np.logaddexp(0, logits / t) - truth * logits / t),
                           bounds=(0.5, 5.0), method="bounded")
    if not best.success:
        raise RuntimeError("Temperature fitting failed")
    return float(best.x)


def bin_index(scores, bins):
    return np.minimum((np.asarray(scores) * bins).astype(int), bins - 1)


def reliability(scores, truth, bins):
    scores, truth = np.asarray(scores), np.asarray(truth)
    indices = bin_index(scores, bins)
    count = np.bincount(indices, minlength=bins)
    confidence = np.bincount(indices, weights=scores, minlength=bins)
    correct = np.bincount(indices, weights=truth, minlength=bins)
    gap = np.abs(confidence - correct)
    occupied = count > 0
    ece = gap.sum() / count.sum() if len(scores) else np.nan
    mce = (gap[occupied] / count[occupied]).max() if len(scores) else np.nan
    brier = np.mean((scores - truth) ** 2) if len(scores) else np.nan
    groups = [dict(bin=j, count=int(n), mean_confidence=confidence[j] / n,
                   accuracy=correct[j] / n) for j, n in enumerate(count) if n]
    return dict(ece=ece, mce=mce, brier=brier, n=len(scores)), groups


def ece_delta_ci(scores, calibrated, truth, image_numbers, image_count, config):
    bins = config["calibration_bins"]
    rng = np.random.default_rng(config["seed"])
    matrices = []
    for values in (scores, calibrated):
        index = image_numbers * bins + bin_index(values, bins)
        matrices.append(np.bincount(index, weights=values - truth,
                                    minlength=image_count * bins).reshape(image_count, bins))
    n = np.bincount(image_numbers, minlength=image_count)
    estimates = []
    for _ in range(config["calibration_bootstrap"]):
        sampled = rng.integers(0, image_count, size=image_count)
        total = n[sampled].sum()
        if total:
            errors = [np.abs(matrix[sampled].sum(axis=0)).sum() / total for matrix in matrices]
            estimates.append(errors[1] - errors[0])
    return np.quantile(estimates, [0.025, 0.975])
