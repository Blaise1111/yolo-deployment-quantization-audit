# Resolution and Quantization Audit of YOLO11n

This study compares detection quality, confidence drift and inference latency across 640, 480 and 320 px inputs. It measures whether lower precision changes low-score ranking or operational detections and whether a temperature fitted on separate images improves INT8 confidence on the fixed evaluation cohort.

## Data and protocol

- The [official COCO 2017 validation distribution](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm) supplies the 5,000 images and instance annotations. The [official Ultralytics YOLO11n release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) supplies the checkpoint. Its SHA-256 is pinned in `configs/coco_audit.yaml`. Archive fingerprints are recorded by `scripts/prepare_assets.py`.
- A seeded permutation of the sorted image IDs (seed 2027) assigns 200 images to `calib_200` and the other 4,800 to `eval_4800`. The fast 1,000-image subset is frozen within `eval_4800` and is used for an export parity gate. The model, scores, calibration strategy and evaluation settings are fixed before the full run. This split isolates the local calibration workflow. The upstream checkpoint was developed for COCO, so this is a benchmark audit rather than validation on a new external dataset.
- The core 3×3 grid uses PyTorch FP32, ONNX Runtime FP32 and ONNX Runtime static INT8 QDQ. The CUDA FP16 ONNX export is a supplementary series. ORT calibrates signed activations and per-output-channel signed weights for Conv/MatMul/Gemm using MinMax on the 200 calibration images. The exported graph emits raw decoded boxes and class probabilities.
- All configurations use the same Python Letterbox with 114 padding, RGB conversion and division by 255. External class-aware NMS uses IoU 0.65, a confidence floor of 0.001 and a global cap of 300 detections per image. The operating audit uses confidence 0.25. Box coordinates are restored using the actual rounded Letterbox dimensions.
- Accuracy uses CUDA for the floating-point configurations and the CPU execution provider for INT8 QDQ. The latency matrix records both CPU and CUDA execution requests where available, including ORT node placement. The CPU has a fixed four-logical-CPU affinity and four intra-op threads. GPU results are from a GTX 1650 Ti in the same Ryzen 5 4600H laptop. CPU timing is an edge proxy and all GPU timing is local laptop timing.

## Results

| Analysis | CSV | Figure |
| --- | --- | --- |
| Paired detection drift and flip attribution | [Backend flips](results/p2_backend_flips_audit.csv) | [Score residuals and flip confidence](figs/p2_fig1_flip_rate_diagnosis.png) |
| COCO scale AP and fixed-FPPI recall | [Core accuracy matrix and FP16 supplement](results/p2_scale_degradation_matrix.csv) | [Latency versus small-object AP and mAP](figs/p2_fig2_small_object_pareto.png) |
| INT8 confidence before and after temperature scaling | [ECE, MCE, Brier score and reliability bins](results/p2_ece_before_after_recal.csv) | [Reliability curves](figs/p2_fig3_reliability_recalibration.png) |
| CPU and GPU latency breakdown | [Stage timing and throughput scaling](results/p2_latency_scaling_breakdown.csv) | [Latency versus input area](figs/p2_fig4_latency_vs_complexity.png) |

On the full 4,800-image evaluation, the absolute PyTorch–ONNX FP32 mAP differences were 5.07e-9 at 320 px, 3.24017031e-7 at 480 px and 5.11e-8 at 640 px. The maximum was approximately 3.24e-7, below the 0.003 parity limit. These values are computed from the [full accuracy CSV](results/p2_scale_degradation_matrix.csv).

### Observed trade-offs

Lower resolution reduced small-object AP and recall across the core grid. INT8 QDQ reduced the ONNX file size and increased CPU end-to-end p95 relative to ORT FP32 at every resolution. The CUDA request for the QDQ graph used mixed CPU and CUDA placement. The provider trace recorded no quantized compute operators executing on CUDA, so those timings characterize that graph's mixed execution. The raw INT8 operating cohort showed higher ECE at 640 and 320 px and lower ECE at 480 px relative to ORT FP32. Temperature scaling reduced ECE on all three fixed cohorts. Maximum calibration error increased at 640 and 320 px. The cost of reducing mean calibration error can include larger error in an individual confidence bin.

### Metric definitions

- **Flips.** At the operating point, boxes are paired by greedy descending IoU ≥ 0.50 within class, using PyTorch at the same resolution as reference. Flip rate is (lost + added) divided by (matched + lost + added). Threshold attribution requires a surviving confidence in [0.20, 0.30] and a same-anchor score crossing. NMS attribution requires evidence that two same-class anchors swap which one suppresses the other under the shared NMS. Other flips are reported as unresolved. The source JSONL records include the pre-NMS candidates needed for this check.
- **Scale.** COCO AP uses the official IoU sweep and default 100-box COCO summary, after the shared 300-box postprocessing cap. At IoU 0.50, a single confidence cutoff maximizes the retained prefix subject to at most 0.5 false positives per evaluated image. Equal scores enter together. Crowd and ignored matches follow COCO rules. Scale recalls divide by the non-crowd ground-truth count in the relevant annotated area band.
- **Calibration.** Correctness uses one-to-one COCO matching at IoU 0.50. Temperature scaling fits binary log loss on the INT8 detections retained in `calib_200`. The same original confidence ≥ 0.25 cohort is compared before and after recalibration on the held-out split. The report checks that the monotone transform preserves score order and ties across the evaluation detections. A paired image-level bootstrap quantifies the change in test ECE. The reported intervals are pointwise per resolution.
- **Timing.** After 50 warmup images, the same 200 evaluation images run in each of three rounds per configuration and device. Decode, Letterbox, synchronized model inference including input/output transfer, and external NMS are timed separately. The total p95 uses each measured end-to-end time directly. The stage stack uses arithmetic means. The memory column is peak host RSS sampled after each timed image in the benchmark process. The CUDA allocation column covers PyTorch's allocator, and ORT graph placement is recorded separately from CUDA memory. The area-only ideal compares 640² / side² to measured end-to-end p95 speedup.

In the public repository, per-image JSONL evidence stays local to the original run. Generate the detection records with the fast and full commands before running the offline CSV check.

## Reproduce

Run from this repository in PowerShell with Python 3.13.16 and uv. Directory `logs/` receives command output. The locked PyTorch build is CUDA 12.8 and the installed ONNX Runtime GPU wheel can also execute on its CPU provider.

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
uv venv --python 3.13.16 *> logs/venv.log
uv pip install -r requirements.txt *> logs/install.log
.venv/Scripts/python.exe scripts/prepare_assets.py *> logs/assets.log
.venv/Scripts/python.exe -m pytest -v tests/test_audit_integrity.py *> logs/pytest.log
.venv/Scripts/python.exe -u scripts/run_full_audit.py --fast-eval-1000 *> logs/fast_full_audit.log
.venv/Scripts/python.exe -u scripts/run_full_audit.py --full *> logs/full_audit.log
.venv/Scripts/python.exe -u scripts/run_full_audit.py --verify-csv *> logs/verify_csv.log
```

The fast command generates local 1,000-image artifacts under `results/fast_eval_1000/` and `figs/fast_eval_1000/`. The full command requires the fast-run FP32 parity gate at every resolution (absolute mAP difference ≤ 0.003). It reuses the latency evidence from the fast run, since the benchmark uses the same 200 images, runtime configuration and local hardware in both modes. Input images and model weights are unnecessary for CSV verification. Keep the per-image detections and candidates, the two ground-truth subsets, `run_manifest.json`, and `latency_logs.csv` together. Use `--verify-csv --results-dir PATH` to regenerate and compare all four reports at an absolute tolerance of `1e-7` from an evidence directory.
