# YOLO deployment evidence after the audit repair

The main analysis uses the frozen COCO val2017 evidence at a8083ab. It is post-hoc and offline. Only the separately labeled 640 px timing check uses new measurements. The [baseline inventory](audit_outputs/baseline/inventory.json) identifies the official COCO download, Ultralytics YOLO11n checkpoint, model-export hashes, current runtime and evidence hashes.

## What was repaired

- FP32 export caches bind the checkpoint SHA-256, export settings and versions, and the actual artifact SHA-256. FP16 and INT8 cache records also bind their upstream FP32 artifact and conversion settings. INT8 additionally fingerprints the ordered calibration images and binds its calibration method. Full runs validate the fast parity record against the current checkpoint, actual FP32 exports, config, annotations, partition membership and fast-image fingerprints. The historical evidence remains readable offline.
- New timing captures bind model fingerprints, ordered image IDs and content hashes, requested execution providers and graph optimization, runtime and driver versions, device, affinity, threads, warm-up, rounds and job order to the CSV digest. Completed and partial captures validate against their manifests. The limited timing check also retains effective ORT provider options and trace hashes. Fast/full reuse requires an exact compatible timing contract.
- Historical timestamps lack a capture-bound model/runtime manifest. [The reconstructed inventory](audit_outputs/yolo/historical_timing_manifest.json) records observed information and leaves the old capture unattested. Saved evidence remains eligible for offline analysis. New capture paths reject it as a reusable timing cache.
- [Timing pairs](audit_outputs/yolo/latency_paired_samples.csv) match each same device, resolution, round and image ID across ORT FP32 and INT8 QDQ. Coverage is one-to-one. The [summary](audit_outputs/yolo/latency_paired_summary.csv) and [round table](audit_outputs/yolo/latency_paired_by_round.csv) report p50/p95 from end-to-end samples. The [stage plot](audit_outputs/yolo/latency_paired_stages.png) uses means of paired stage differences. The instrument groups transfer, model execution and synchronization into one inference stage.
- Partition checks confirm disjoint 200-image calibration and 4,800-image evaluation sets. The 1,000-image export parity gate is nested inside the evaluation cohort. No model or cutoff selection from that subset is specified by the original protocol. This audit therefore reports the full historical cohort. It remains internal to COCO.

## What the evidence shows

Historic 640 px CPU p95 was 70.65 ms for ORT FP32 and 112.59 ms for QDQ. The mean paired end-to-end increase was 38.47 ms, of which 38.23 ms was in the joint inference/transfer/synchronization stage. CUDA QDQ used mixed placement. The [provider summary](audit_outputs/yolo/historical_placement.csv) reports node counts and an operator allowlist. It cannot attribute elapsed time to individual kernels or establish native CUDA INT8 compute.

The [FP32-first](audit_outputs/timing_ab/paired_smoke_summary.csv) and [INT8-first](audit_outputs/timing_ba/paired_smoke_summary.csv) checks each used the same 24 images, two timed rounds per device, and eight warmups per job on the local Ryzen 5 4600H four-thread CPU proxy and GTX 1650 Ti. QDQ was slower at CPU p95 in both orders. The two directories contain the actual CSV, manifests and ORT traces. Small-sample latency is a diagnostic under these conditions.

The [fixed-cohort table](audit_outputs/yolo/calibration_fixed_cohort.csv) holds the original score ≥0.25 detection set constant. Temperature is estimated on the 200 calibration images and evaluated on the held-out set. Fixed-cohort ECE declined at all three resolutions, with pointwise, image-paired intervals below zero. At 640 px ECE moved from 14.80% to 13.86%, while MCE rose from 19.54% to 21.55%. Support, Brier scores and intervals are in the table.

The [operating-point table with threshold equivalence](audit_outputs/yolo_threshold_equivalence/calibration_rethresholded.csv) compares the raw 0.25 threshold, 0.25 after temperature scaling, and direct raw-score filtering at `sigmoid(T × logit(0.25))`. Positive-temperature scaling preserves score order. The direct raw and calibrated filters retained identical ordered (image ID, detection ID) pairs at 320, 480 and 640 px. The table records each selection's SHA-256. COCO one-to-one class matching and crowd rules are shared across rows.

At 640 px the equivalent raw threshold is approximately **0.3020835**. The comparison used the unrounded threshold. Both filters retained **21,722** boxes, with 16,514 TP, 4,503 FP and 18,458 FN. Relative to raw 0.25, precision went from 73.91% to 78.57%, and recall from 50.33% to 47.22%. Adjusting the raw threshold reproduces this operating-point change. The fixed-cohort ECE comparison above addresses how well scores represent correctness for the same detections. The operating-point comparison is limited to saved post-NMS boxes, and the original unresolved flip classifications are kept.

## Reproduce and verify

Install the locked requirements as described in [the README](README.md). Repo-tracked CSVs suffice for the timing analysis. Calibration and reports-only additionally require the prior per-image JSONL files, which stay local. Obtain the official images, annotations and pretrained weights using the README if you need to generate predictions or run the new timing check. The offline commands use a read-only source evidence directory.

~~~powershell
.venv/Scripts/python.exe scripts/run_full_audit.py --verify-csv --results-dir results
.venv/Scripts/python.exe scripts/reaudit_saved_evidence.py --results-dir results --output audit_outputs/yolo_recomputed
.venv/Scripts/python.exe scripts/reaudit_saved_evidence.py --results-dir results --calibration-only --output audit_outputs/yolo_threshold_equivalence
.venv/Scripts/python.exe scripts/run_full_audit.py --reports-only --results-dir results --output-dir audit_outputs/reports_check
.venv/Scripts/python.exe scripts/timing_smoke.py --output audit_outputs/timing_ab --with-cuda
.venv/Scripts/python.exe scripts/timing_smoke.py --output audit_outputs/timing_ba --order int8-first --with-cuda
.venv/Scripts/python.exe -m pytest -q
~~~

The [legacy checks](audit_outputs/baseline/commands.json), [timing check commands](audit_outputs/smoke_commands.json) and [offline input manifest](audit_outputs/yolo/analysis_manifest.json) distinguish measurement from recomputation. Repeating the first timing command reuses only its verified capture. An intentional reverse-order request against the same directory failed as expected with incompatible job order.

Local command and test outcomes are listed in the [validation record](audit_outputs/validation_commands.json).

The [prepublication validation record](audit_outputs/prepublish_validation.json) records the follow-up commands, their results and remaining verification limits.

## Status

- **Repaired** · Explicit provenance and checked reuse, phase-paired timing, and separate fixed-cohort and operational calibration estimands.
- **Measured locally** · An attested 640 px timing check on CPU and CUDA, with both backend orders.
- **Offline recomputation** · Full historical pairing, held-out calibration and post-NMS operating metrics. The historical p95 values and public summary checks remain reproducible from the recorded inputs.
- **Unverified** · Capture-bound provenance for old timing, kernel-specific causation, a new full prediction run and external-device generalization. The originally saved inputs were sufficient for these offline questions.

## Application evidence copy for review

I audited YOLO11n on public COCO data by pairing timing samples and separating fixed-cohort confidence calibration from post-NMS threshold choice. At 640 px the historical CPU p95 was 70.65 ms for FP32 and 112.59 ms for QDQ, and small checks on the same laptop reproduced the direction in both execution orders. For the same saved INT8 detection cohort, temperature scaling decreased ECE from 14.80% to 13.86%. Reapplying 0.25 after calibration retained the same 21,722 post-NMS boxes as applying approximately 0.3020835 to the original scores, reproducing the precision and recall change. [Timing comparison](audit_outputs/yolo/latency_paired_summary.csv) · [Threshold comparison](audit_outputs/yolo_threshold_equivalence/calibration_rethresholded.csv)

Publication link pending local review.
