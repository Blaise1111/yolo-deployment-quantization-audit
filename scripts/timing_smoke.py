"""Measure a separately attested 640-px timing check without prediction regeneration."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch

from src.coco_split import ROOT, coco_inputs, load_config
from src.latency_bench import benchmark
from src.offline_reaudit import summarize_pairs, timing_pairs
from src.reports import csv_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--order",choices=("fp32-first","int8-first"),default="fp32-first")
    parser.add_argument("--with-cuda",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    historical=(ROOT/"results").resolve()
    if output==historical or historical in output.parents:
        parser.error("Pick a separate local output directory")
    if args.with_cuda and not torch.cuda.is_available():
        parser.error("CUDA timing requested but CUDA is unavailable")
    torch.set_num_interop_threads(1)
    config=load_config()
    config.update(latency_images=24,latency_rounds=2,warmups=8)
    source,images,partitions=coco_inputs(config)
    categories=sorted(source["categories"],key=lambda r:r["id"])
    runtimes=["onnx_fp32","onnx_int8"]
    if args.order=="int8-first":
        runtimes.reverse()
    devices=["cpu","cuda"] if args.with_cuda else ["cpu"]
    jobs=[(640,backend,device) for device in devices for backend in runtimes]
    output.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    path=output/"latency_logs.csv"
    benchmark(path,jobs,images,partitions["eval_1000"],categories,config)
    rows=pd.read_csv(path,float_precision="round_trip")
    csv_write(summarize_pairs(timing_pairs(rows),["resolution","device"]),
              output/"paired_smoke_summary.csv")


if __name__=="__main__":
    main()
