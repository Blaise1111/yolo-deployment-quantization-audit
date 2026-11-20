"""Post-hoc timing and calibration audits from saved public COCO evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.coco_split import ROOT, digest, split_ids, write_json
from src.detections import read_records
from src.offline_reaudit import (calibration_audit, placement, summarize_pairs,
                                timing_pairs, validate_partitions)
from src.reports import csv_write, latency_report
from src.timing_evidence import PHASES, chosen_images, load, path_for


def draw(pairs, metrics, path):
    fig, axes=plt.subplots(1,2,figsize=(10,4.2),layout="constrained")
    sample=pairs[pairs.resolution.eq(640)]
    groups=list(sample.groupby("device"))
    ax=axes[0]
    ax.boxplot([f.total_ms_paired_delta for _,f in groups],tick_labels=[k.upper() for k,_ in groups],
               showfliers=False)
    ax.axhline(0,color="black",lw=.7)
    ax.set(title="640 px · paired end-to-end difference",ylabel="INT8 − FP32 (ms)")
    ax=axes[1]
    positions=np.arange(len(metrics))
    for stage,offset in zip(PHASES,np.linspace(-.28,.28,len(PHASES))):
        ax.bar(positions+offset,metrics[f"mean_{stage}_paired_delta"],width=.18,label=stage)
    ax.set_xticks(positions,[f"{r.resolution}\n{r.device.upper()}" for r in metrics.itertuples()])
    ax.axhline(0,color="black",lw=.7)
    ax.set(title="Mean paired difference by measured stage",ylabel="INT8 − FP32 (ms)")
    ax.legend(frameon=False)
    fig.savefig(path,dpi=240)
    plt.close(fig)


def calibration_reports(root, output, metadata, cal_ids, ids, inputs):
    fixed,operating=[],[]
    for size in metadata["config"]["resolutions"]:
        run=f"detections_onnx_int8_{size}.jsonl"
        cal=f"calib_int8_{size}.jsonl"
        for name,expected in ((run,ids),(cal,cal_ids)):
            csv_path=root/name
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing saved calibration evidence: {name}")
            inputs[name]=digest(csv_path)
            provenance=csv_path.with_suffix(".provenance.json")
            if provenance.is_file():
                record=json.loads(provenance.read_text(encoding="utf-8"))
                if (record["ids"]!=expected or record["config"]!=metadata["config"]
                        or record["device"]!=metadata["accuracy_backends"]["onnx_int8"]
                        or record["annotations_sha256"]!=metadata["annotations_sha256"]):
                    raise ValueError(f"Prediction provenance differs: {name}")
                inputs[provenance.name]=digest(provenance)
        gt,cal_gt=(json.loads((root/f"{s}_ground_truth.json").read_text(encoding="utf-8"))
                   for s in ("eval","calib"))
        for annotations,partition in ((gt,ids),(cal_gt,cal_ids)):
            if sorted(r["id"] for r in annotations["images"])!=partition:
                raise ValueError("Saved ground truth differs from manifest partitions")
        metrics, op, common=calibration_audit(metadata["config"],gt,cal_gt,
                       read_records(root/run,ids),read_records(root/cal,cal_ids),ids)
        fixed.extend(dict(resolution=size,**common,**row) for row in metrics)
        operating.extend(dict(resolution=size,**common,**row) for row in op)
        print(f"Completed fixed-cohort and re-thresholded calibration at {size}px",flush=True)
    csv_write(pd.DataFrame(fixed),output/"calibration_fixed_cohort.csv")
    csv_write(pd.DataFrame(operating),output/"calibration_rethresholded.csv")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--results-dir",type=Path,default=ROOT/"results")
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--calibration-only",action="store_true",help="Recompute saved calibration evidence without timing")
    args=cli.parse_args()
    evidence,output=args.results_dir.resolve(),args.output.resolve()
    historical=(ROOT/"results").resolve()
    if output in (evidence,historical) or evidence in output.parents or historical in output.parents:
        cli.error("Use an independent audit output directory")
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((evidence/"run_manifest.json").read_text(encoding="utf-8"))
    if metadata["eval_partition"]!="eval_4800":
        cli.error("Use the full-cohort run manifest, not fast-only results")
    parts=metadata["partitions"]
    cal_ids,ids=validate_partitions(parts)
    if tuple(parts[k] for k in ("calib_200","eval_4800","eval_1000"))!=split_ids(
        cal_ids+ids,metadata["config"]["seed"],metadata["config"]["calib_count"]):
        raise ValueError("Partition membership differs from the documented seed")
    inputs={"run_manifest.json":digest(evidence/"run_manifest.json")}
    if not args.calibration_only:
        inputs["latency_logs.csv"]=digest(evidence/"latency_logs.csv")
        n_pairs=timing_reports(evidence,output,metadata,parts,inputs)
    audit=dict(mode="post_hoc_offline", baseline="a8083ab", config=metadata["config"],
               calibration_partition=cal_ids,evaluation_partition=ids,
               rethreshold_stage="after saved NMS and max_det, before threshold selection",
               threshold_equivalence="sigmoid(T * logit(q)); exact saved image and detection ID comparison",
               matching="COCO class-wise one-to-one IoU .50; ignore/crowd per COCO",
               input_sha256=inputs, calibration_status="pending")
    try:
        for name in ("eval_ground_truth.json","calib_ground_truth.json"):
            inputs[name]=digest(evidence/name)
        calibration_reports(evidence,output,metadata,cal_ids,ids,inputs)
    except FileNotFoundError as exc:
        audit.update(calibration_status="blocked_missing_evidence",missing=Path(exc.filename or str(exc)).name)
        write_json(output/"analysis_manifest.json",audit)
        cli.exit(2,f"Calibration blocked: {exc}\n")
    audit["calibration_status"]="complete"
    write_json(output/"analysis_manifest.json",audit)
    if not args.calibration_only:
        print(f"Recomputed {n_pairs} matched timing pairs",flush=True)


def timing_reports(evidence, output, metadata, parts, inputs):
    timing=evidence/"latency_logs.csv"
    latency_report(evidence)
    if path_for(timing).is_file():
        raw_timing,attestation=load(timing)
        status=attestation["origin"]
        inputs[path_for(timing).name]=digest(path_for(timing))
    else:
        raw_timing=pd.read_csv(timing,float_precision="round_trip")
        status="historical_offline_only_unattested"
    order=chosen_images(parts["eval_1000"],metadata["config"])
    if any(group.image_id.tolist()!=order for _,group in raw_timing.groupby(
            ["resolution","backend","device","round"],sort=False)):
        raise ValueError("Recorded order differs from the historical benchmark selection")
    pairs=timing_pairs(raw_timing)
    metrics=summarize_pairs(pairs,["resolution","device"])
    csv_write(pairs,output/"latency_paired_samples.csv")
    csv_write(metrics,output/"latency_paired_summary.csv")
    csv_write(summarize_pairs(pairs,["resolution","device","round"]),output/"latency_paired_by_round.csv")
    csv_write(placement(raw_timing),output/"historical_placement.csv")
    draw(pairs,metrics,output/"latency_paired_stages.png")
    write_json(output/"historical_timing_manifest.json",dict(
        source=status, sha256=inputs["latency_logs.csv"], units="ms",
        image_order=order, rounds=metadata["config"]["latency_rounds"],
        warmups_declared_unattested=metadata["config"]["warmups"],
        hardware_declared_unattested=metadata["hardware"],
        row_threads=sorted(set(int(x) for x in raw_timing.host_threads)),
        row_affinity=sorted(set(raw_timing.host_affinity.astype(str))),
        model_export_hashes_bound_to_capture=False if status=="historical_offline_only_unattested" else True,
        runtime_and_graph_optimization_attested=status!="historical_offline_only_unattested",
        fast_membership="1000-image subset of eval_4800"))
    return len(pairs)


if __name__=="__main__":
    main()
