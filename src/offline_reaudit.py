"""Offline estimands for recorded timing pairs and post-NMS calibration."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .coco_metrics import evaluate
from .flips_and_ece import ece_delta_ci, reliability, temperature_fit, transform
from .reports import cohort
from .timing_evidence import PHASES, validate_rows


def timing_pairs(rows):
    validate_rows(rows)
    keys=["resolution","device","round","image_id"]
    columns=[name+"_ms" for name in (*PHASES,"total")]
    left=rows[rows.backend.eq("onnx_fp32")].set_index(keys)[columns]
    right=rows[rows.backend.eq("onnx_int8")].set_index(keys)[columns]
    if left.empty or len(left.index.symmetric_difference(right.index)):
        raise ValueError("ORT FP32 and INT8 timing must have complete one-to-one sample coverage")
    right=right.loc[left.index]
    return pd.concat([left.add_suffix("_fp32"),right.add_suffix("_int8"),
                      (right-left).add_suffix("_paired_delta")],axis=1).reset_index()


def summarize_pairs(pairs, keys):
    rows=[]
    for identity, group in pairs.groupby(keys):
        row=dict(zip(keys,identity),n_pairs=len(group),pair_coverage=1.0,units="ms")
        for name in ("fp32","int8","paired_delta"):
            t=group["total_ms_"+name]
            row.update({f"total_p50_{name}":t.quantile(.5),f"total_p95_{name}":t.quantile(.95)})
        for stage in (*PHASES,"total"):
            row[f"mean_{stage}_paired_delta"]=group[f"{stage}_ms_paired_delta"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def placement(rows):
    result=[]
    for (size,backend,device), group in rows.groupby(["resolution","backend","device"]):
        fields={}
        for key in ("node_placement","int8_compute_placement"):
            value=group[key].unique()
            if len(value)!=1:
                raise ValueError(f"Inconsistent recorded {key}")
            fields[key]=json.dumps(json.loads(value[0]),sort_keys=True)
        result.append(dict(resolution=size,backend=backend,device=device,**fields,
                           source="historical CSV node counters; operator allowlist, not kernel timing"))
    return pd.DataFrame(result)


def validate_partitions(parts):
    c, e, f = (parts[k] for k in ("calib_200","eval_4800","eval_1000"))
    if (len(c)!=200 or len(e)!=4800 or len(f)!=1000
            or any(len(x)!=len(set(x)) for x in (c,e,f)) or set(c)&set(e) or not set(f)<=set(e)):
        raise ValueError("COCO partitions violate the 200/4800 disjoint split or nested fast gate")
    return sorted(c),sorted(e)


def operating_row(scores, truth, cut, gt_count, total_candidates):
    mask=scores>=cut
    n=int(mask.sum())
    tp=int(truth[mask].sum())
    if not 0<=tp<=gt_count or total_candidates<n:
        raise ValueError("Invalid operating cohort support")
    return dict(detections_total=total_candidates, matched_evaluable=n, ignored=total_candidates-n,
                tp=tp, fp=n-tp, fn=gt_count-tp, precision=tp/n if n else np.nan,
                recall=tp/gt_count if gt_count else np.nan, ground_truth_non_crowd=gt_count)


def equivalent_raw_threshold(cut, temperature):
    if not 0 < cut < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Threshold must be in (0, 1) and temperature must be finite and positive")
    return float(expit(temperature * logit(cut)))


def post_nms_cohorts(records, cut, temperature):
    raw_cut = equivalent_raw_threshold(cut, temperature)
    original, adjusted, equivalent = [], [], []
    for image_id, record in sorted(records.items()):
        detections = record["detections"]
        keys = [(image_id, row[0]) for row in detections]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate saved detection IDs for image {image_id}")
        scores = np.asarray([row[2] for row in detections], dtype=np.float64)
        for target, mask in ((original, scores >= cut),
                             (adjusted, transform(scores, temperature) >= cut),
                             (equivalent, scores >= raw_cut)):
            target.extend(key for key, keep in zip(keys, mask) if keep)
    if adjusted != equivalent:
        raise ValueError("Temperature and equivalent raw thresholds selected different detection IDs")
    return raw_cut, original, adjusted, equivalent


def calibration_audit(config, gt, cal_gt, records, calibration, ids):
    cut=config["operating_confidence"]
    _,cal_status,_,_=evaluate(cal_gt,calibration,config["max_det"],full=False)
    cal_scores,cal_truth,_=cohort(cal_status,config)
    temperature=temperature_fit(cal_scores,cal_truth)
    _,status,anns,_=evaluate(gt,records,config["max_det"],full=False)
    raw=status[:,0]
    adjusted=transform(raw,temperature)
    truth=(status[:,1]>0).astype(float)
    image_index=np.searchsorted(ids,status[:,2].astype(int))
    unique=np.unique([x[2] for entry in records.values() for x in entry["detections"]])
    if np.any(np.diff(transform(unique,temperature))<=0):
        raise ValueError("Post-NMS subset matching requires the full detection rank order to be preserved")
    keep=raw>=cut
    lo,hi=ece_delta_ci(raw[keep],adjusted[keep],truth[keep],image_index[keep],len(ids),config)
    raw_metrics,_=reliability(raw[keep],truth[keep],config["calibration_bins"])
    fitted_metrics,_=reliability(adjusted[keep],truth[keep],config["calibration_bins"])
    gt_count=sum(not obj["iscrowd"] for obj in anns.values())
    raw_cut, original_ids, adjusted_ids, equivalent_ids = post_nms_cohorts(records, cut, temperature)
    if not np.array_equal(adjusted >= cut, raw >= raw_cut):
        raise ValueError("Equivalent thresholds differ on COCO-evaluable detections")

    def operation(method, scores, threshold, selected):
        identity_sha256 = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
        return dict(method=method, applied_threshold=threshold, selected_ids_sha256=identity_sha256,
                    **operating_row(scores, truth, threshold, gt_count, len(selected)))
    return (
        [dict(method="raw",**raw_metrics),
         dict(method="temperature",**fitted_metrics,delta_ece=fitted_metrics["ece"]-raw_metrics["ece"],
              delta_ece_ci_low=lo,delta_ece_ci_high=hi)],
        [operation("raw_threshold", raw, cut, original_ids),
         operation("reapply_threshold_after_temperature", adjusted, cut, adjusted_ids),
         operation("equivalent_raw_threshold", raw, raw_cut, equivalent_ids)],
        dict(temperature=temperature,calibration_detector_support=len(cal_scores),threshold=cut))
