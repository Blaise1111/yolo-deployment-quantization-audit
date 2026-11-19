"""Publication figures for paired detection drift, scale, calibration and timing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .reports import REPORTS

LABELS = dict(torch_fp32="PyTorch FP32", onnx_fp32="ORT FP32",
              onnx_fp16="ORT FP16", onnx_int8="ORT INT8 QDQ")
COLORS = dict(torch_fp32="#355777", onnx_fp32="#5c8068",
              onnx_fp16="#9d673a", onnx_int8="#943e57")


def save(fig, output, filename):
    fig.savefig(output/filename, dpi=300, facecolor="white")
    plt.close(fig)


def drift_plot(table, details, output):
    residuals, flips = details
    sizes = sorted(table.resolution.unique(), reverse=True)
    rng = np.random.default_rng(2027)
    fig, axes = plt.subplots(len(sizes), 3, figsize=(14.8, 8.5), layout="constrained",
                             gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    reasons = [("threshold_edge", "Threshold crossing", "#4f796c"),
               ("nms_reversal", "NMS reversal", "#8e4d62"),
               ("unresolved", "Unresolved", "#8c9199")]
    for row, size in enumerate(sizes):
        scatter, histogram, attribution = axes[row]
        for backend in [key for key in LABELS if key != "torch_fp32" and (size,key) in residuals]:
            points = np.asarray(residuals[size, backend]).reshape(-1, 2)
            selected = rng.choice(len(points), size=min(len(points), 1800), replace=False)
            scatter.scatter(points[selected, 0], points[selected, 1], s=3, alpha=.3,
                            color=COLORS[backend], label=LABELS[backend], rasterized=True)
            scores = [score for score, _, _ in flips[size, backend]]
            if scores:
                histogram.hist(scores, bins=np.linspace(.25, 1, 31), histtype="step", lw=1.6,
                               color=COLORS[backend], label=LABELS[backend])
        scatter.axhline(0, color="#888888", lw=.8)
        scatter.set(title=f"{size} px | matched detections", ylabel="Variant score − reference")
        histogram.axvspan(.25, .30, color="#cfcfcf", alpha=.35)
        histogram.set(title=f"{size} px | added and lost detections", ylabel="Box count")
        variants = table[(table.resolution == size) & (table.backend != "torch_fp32")]
        labels = [f"{LABELS[v.backend]} ({v.flip_rate:.1%})" for v in variants.itertuples()]
        left = np.zeros(len(variants))
        for key, label, color in reasons:
            values = variants[key + "_count"].to_numpy()
            attribution.barh(labels, values, left=left, color=color, label=label)
            left += values
        attribution.invert_yaxis()
        attribution.tick_params(axis="y", labelsize=7)
        attribution.set(title=f"{size} px | flip attribution", xlabel="Added and lost boxes")
    axes[0, 2].legend(frameon=False, fontsize=7, loc="lower right")
    axes[-1, 0].set_xlabel("PyTorch confidence")
    axes[-1, 1].set_xlabel("Confidence of surviving box")
    axes[0, 0].legend(frameon=False, fontsize=8, markerscale=2)
    fig.suptitle("Backend drift at confidence ≥ 0.25")
    save(fig, output, "p2_fig1_flip_rate_diagnosis.png")


def pareto_points(x, y):
    best = -np.inf
    keep = []
    for i in np.lexsort((-y, x)):
        if y[i] > best:
            keep.append(i)
            best = y[i]
    return keep


def pareto_plot(accuracy, latency, output):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8), layout="constrained")
    combined = latency.merge(accuracy, on=["resolution","backend"])
    # Accuracy is evaluated on each backend's documented device. CUDA QDQ can use
    # a different mixed-placement graph and is shown only in the latency figure.
    combined = combined[~((combined.backend == "onnx_int8") & (combined.device == "cuda"))]
    markers = dict(cpu="o", cuda="^")
    panels = [("ap_small", "AP for small objects"), ("map", "COCO mAP"),
              ("recall_small_fppi", "Small-object recall at FPPI ≤ 0.5")]
    for ax, (metric, title) in zip(axes, panels):
        for device, rows in combined.groupby("device"):
            x, y = rows.total_p95_ms.to_numpy(), rows[metric].to_numpy()
            keep = pareto_points(x, y)
            ax.plot(x[keep], y[keep], color="#818181", linestyle="--", linewidth=1)
            for item in rows.itertuples():
                ax.scatter(item.total_p95_ms, getattr(item, metric), marker=markers[device], s=45,
                           color=COLORS[item.backend], edgecolors="white", linewidth=.35)
                if item.backend in ("onnx_int8", "torch_fp32"):
                    ax.annotate(str(item.resolution), (item.total_p95_ms, getattr(item, metric)),
                                xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set(title=title, xlabel="End-to-end p95 (ms)", ylabel=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.18)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", linestyle="", color=color, label=LABELS[key])
              for key, color in COLORS.items() if key in set(combined.backend)]
    legend.extend(Line2D([0], [0], marker=marker, linestyle="", color="#555555", label=device.upper())
                  for device, marker in markers.items() if device in set(combined.device))
    axes[-1].legend(handles=legend, fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Resolution and precision trade-offs")
    save(fig, output, "p2_fig2_small_object_pareto.png")


def calibration_plot(table, output):
    sizes = sorted(table.resolution.unique(), reverse=True)
    fig, axes = plt.subplots(1, len(sizes), figsize=(12.4, 4.2), layout="constrained",
                             sharex=True, sharey=True)
    groups = (("onnx_fp32","raw","ORT FP32","#355777"),
              ("onnx_int8","raw","INT8 raw","#943e57"),
              ("onnx_int8","temperature","INT8 temperature","#47836e"))
    for ax, size in zip(axes, sizes):
        ax.plot([0,1], [0,1], color="#747474", linestyle="--", linewidth=1)
        selected = table[table.resolution == size]
        for backend, method, label, color in groups:
            data = selected[(selected.backend == backend) & (selected.method == method)]
            bins = data[data.kind == "bin"].sort_values("bin")
            error = data.loc[data.kind.eq("metric"), "ece"].iloc[0]
            ax.plot(bins.mean_confidence, bins.accuracy, marker="o", ms=4, color=color,
                    label=f"{label} ({error:.3f})")
        ax.set(title=f"{size} px", xlabel="Mean confidence", xlim=(0,1), ylim=(0,1))
        ax.grid(alpha=.14)
    axes[0].set_ylabel("Observed correctness at IoU 0.50")
    axes[-1].legend(title="ECE", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle("Reliability on detections retained at the original operating threshold")
    save(fig, output, "p2_fig3_reliability_recalibration.png")


def latency_plot(table, output):
    devices = sorted(table.device.unique())
    fig, axes = plt.subplots(2, len(devices), figsize=(7*len(devices), 8.2),
                             squeeze=False, layout="constrained")
    stages = dict(decode="#a9bbcd", preprocess="#8aa491", inference="#355777",
                  postprocess="#b27353")
    for col, device in enumerate(devices):
        upper, lower = axes[:, col]
        subset = table[table.device == device].sort_values(["backend","resolution"], ascending=[True,False])
        x = np.arange(len(subset))
        base = np.zeros(len(x))
        for stage, color in stages.items():
            values = subset[f"{stage}_mean_ms"].to_numpy()
            upper.bar(x, values, bottom=base, color=color, label=stage.replace("postprocess", "NMS postprocess"))
            base += values
        labels = [f"{LABELS[r.backend]}\n{r.resolution}" for r in subset.itertuples()]
        upper.set(title=f"{device.upper()} | mean stage time", ylabel="Milliseconds")
        upper.set_xticks(x, labels, fontsize=7, rotation=45, ha="right")
        for backend, rows in subset.groupby("backend"):
            order = rows.sort_values("resolution", ascending=False)
            lower.plot(order.resolution, order.total_p95_speedup_vs_640,
                       marker="o", color=COLORS[backend], label=LABELS[backend])
        sizes = np.sort(subset.resolution.unique())[::-1]
        lower.plot(sizes, (640/sizes)**2, color="#777777", linestyle="--", label="Area-only ideal")
        lower.set(title=f"{device.upper()} | end-to-end speedup",
                  xlabel="Input side (px)", ylabel="640 px p95 / current p95")
        lower.set_xticks(sizes)
        lower.invert_xaxis()
        lower.grid(axis="y", alpha=.18)
        lower.legend(frameon=False, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Pipeline latency and input-area scaling")
    save(fig, output, "p2_fig4_latency_vs_complexity.png")


def create_plots(reports, details, output):
    output.mkdir(parents=True, exist_ok=True)
    flips, ranking, calibrated, timing = (reports[key] for key in REPORTS)
    drift_plot(flips, details, output)
    pareto_plot(ranking, timing, output)
    calibration_plot(calibrated, output)
    latency_plot(timing, output)
