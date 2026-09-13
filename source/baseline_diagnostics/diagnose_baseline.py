

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASSES = (3, 4, 5, 6)
TRANSITIONS = ((3, 2), (4, 3), (5, 4), (6, 5), (6, 8))
OVERLAP_BINS = (
    ("1-3", 1, 3),
    ("4-7", 4, 7),
    ("8", 8, 8),
    ("9-11", 9, 11),
    (">=12", 12, 10_000),
)


def load_npy(path: Path) -> np.ndarray:
    """Load a .npy, repairing a UTF-8-mangled magic byte (0x93 -> U+FFFD)."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbf\xbdNUMPY"):
        raw = b"\x93" + raw[3:]
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{path} did not contain an ndarray")
    return arr


def iou_binary(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    p = pred == cls
    g = gt == cls
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return np.nan
    return float(inter / union)


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def depth_stats(mask: np.ndarray) -> tuple[float, float, float, int]:
    """mask is a 2D (H, W) boolean slice. Axis 0 is depth/Y."""
    if not mask.any():
        return (np.nan, np.nan, np.nan, 0)
    ys = np.where(mask)[0].astype(np.float64)
    return (float(ys.mean()), float(np.median(ys)), float(ys.std()), int(ys.size))


def quartile_slices(n: int) -> list[tuple[str, slice]]:
    edges = np.linspace(0, n, 5, dtype=int)
    labels = ["Q1_first_25", "Q2", "Q3", "Q4_last_25"]
    return [(labels[i], slice(int(edges[i]), int(edges[i + 1]))) for i in range(4)]


def save_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            out = []
            for v in row:
                if v is None:
                    out.append("")
                elif isinstance(v, float):
                    if np.isnan(v):
                        out.append("")
                    else:
                        out.append(f"{v:.8f}")
                else:
                    out.append(str(v))
            f.write(",".join(out) + "\n")


def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_lines(path, xs, series, xlabel, ylabel, title, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for name, ys, kw in series:
        ax.plot(xs, ys, label=name, **kw)
    style_axes(ax, xlabel, ylabel, title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--input_dir",
    default="../test_logs_volume",
    help="Folder with prediction_volume.npy / ground_truth_volume.npy / prediction_count.npy",
)
    parser.add_argument(
        "--output_dir",
        default=".",
    )
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pred = load_npy(inp / "prediction_volume.npy")
    gt = load_npy(inp / "ground_truth_volume.npy")
    count = load_npy(inp / "prediction_count.npy")

    if pred.shape != gt.shape or pred.shape != count.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} gt={gt.shape} count={count.shape}")

    z_n, y_n, x_n = pred.shape
    slice_vox = y_n * x_n
    z = np.arange(z_n)
    abs_crossline = z + 809  # original survey index of the spatial test split

    # ------------------------------------------------------------------
    # 1. Crossline class-fraction profiles
    # ------------------------------------------------------------------
    gt_frac = {c: np.empty(z_n, np.float64) for c in CLASSES}
    pr_frac = {c: np.empty(z_n, np.float64) for c in CLASSES}
    for i in range(z_n):
        g = gt[i]
        p = pred[i]
        for c in CLASSES:
            gt_frac[c][i] = float((g == c).mean())
            pr_frac[c][i] = float((p == c).mean())

    save_csv(
        out / "crossline_profiles.csv",
        ["z", "abs_crossline"]
        + [f"gt_frac_{c}" for c in CLASSES]
        + [f"pred_frac_{c}" for c in CLASSES],
        [
            [i, int(abs_crossline[i])]
            + [gt_frac[c][i] for c in CLASSES]
            + [pr_frac[c][i] for c in CLASSES]
            for i in range(z_n)
        ],
    )

    colors = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728"}
    series = []
    for c in CLASSES:
        series.append((f"GT {c}", gt_frac[c], {"color": colors[c], "lw": 1.8}))
        series.append(
            (f"Pred {c}", pr_frac[c], {"color": colors[c], "lw": 1.4, "ls": "--"})
        )
    plot_lines(
        fig_dir / "crossline_class_fractions.png",
        z,
        series,
        "test crossline z (0 = survey 809, 141 = survey 950)",
        "voxel fraction",
        "Class 3–6 occupancy along the test volume",
        ylim=(0, None),
    )

    # ------------------------------------------------------------------
    # 2. Depth / Y profiles
    # ------------------------------------------------------------------
    depth_rows = []
    depth_store = {
        c: {
            "gt_mean": np.full(z_n, np.nan),
            "gt_med": np.full(z_n, np.nan),
            "gt_std": np.full(z_n, np.nan),
            "pr_mean": np.full(z_n, np.nan),
            "pr_med": np.full(z_n, np.nan),
            "pr_std": np.full(z_n, np.nan),
            "gt_n": np.zeros(z_n, np.int64),
            "pr_n": np.zeros(z_n, np.int64),
        }
        for c in (4, 5, 6)
    }
    for i in range(z_n):
        row = [i, int(abs_crossline[i])]
        for c in (4, 5, 6):
            gmean, gmed, gstd, gn = depth_stats(gt[i] == c)
            pmean, pmed, pstd, pn = depth_stats(pred[i] == c)
            st = depth_store[c]
            st["gt_mean"][i] = gmean
            st["gt_med"][i] = gmed
            st["gt_std"][i] = gstd
            st["pr_mean"][i] = pmean
            st["pr_med"][i] = pmed
            st["pr_std"][i] = pstd
            st["gt_n"][i] = gn
            st["pr_n"][i] = pn
            row.extend([gmean, gmed, gstd, gn, pmean, pmed, pstd, pn])
        depth_rows.append(row)

    depth_header = ["z", "abs_crossline"]
    for c in (4, 5, 6):
        depth_header += [
            f"gt_{c}_mean_y",
            f"gt_{c}_median_y",
            f"gt_{c}_std_y",
            f"gt_{c}_n",
            f"pred_{c}_mean_y",
            f"pred_{c}_median_y",
            f"pred_{c}_std_y",
            f"pred_{c}_n",
        ]
    save_csv(out / "depth_profiles.csv", depth_header, depth_rows)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, c in zip(axes, (4, 5, 6)):
        ax.plot(z, depth_store[c]["gt_mean"], color=colors[c], lw=1.8, label=f"GT {c} mean Y")
        ax.plot(
            z,
            depth_store[c]["pr_mean"],
            color=colors[c],
            lw=1.4,
            ls="--",
            label=f"Pred {c} mean Y",
        )
        ax.fill_between(
            z,
            depth_store[c]["gt_mean"] - depth_store[c]["gt_std"],
            depth_store[c]["gt_mean"] + depth_store[c]["gt_std"],
            color=colors[c],
            alpha=0.12,
        )
        style_axes(ax, "", "mean depth Y (0=top)", f"Class {c} depth along test")
        ax.legend(fontsize=8)
        ax.invert_yaxis()
    axes[-1].set_xlabel("test crossline z")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_profiles.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Error / transition profiles
    # ------------------------------------------------------------------
    overall_err = np.empty(z_n, np.float64)
    cls_err = {c: np.full(z_n, np.nan) for c in CLASSES}
    trans_rate = {pair: np.full(z_n, np.nan) for pair in TRANSITIONS}
    trans_mass = {pair: np.zeros(z_n, np.float64) for pair in TRANSITIONS}

    for i in range(z_n):
        g = gt[i]
        p = pred[i]
        wrong = p != g
        overall_err[i] = float(wrong.mean())
        for c in CLASSES:
            m = g == c
            n = int(m.sum())
            if n:
                cls_err[c][i] = float((p[m] != c).mean())
        for a, b in TRANSITIONS:
            m = g == a
            n = int(m.sum())
            if n:
                trans_rate[(a, b)][i] = float((p[m] == b).mean())
            trans_mass[(a, b)][i] = float((m & (p == b)).sum()) / slice_vox

    err_header = (
        ["z", "abs_crossline", "overall_error"]
        + [f"error_true_{c}" for c in CLASSES]
        + [f"rate_{a}_to_{b}" for a, b in TRANSITIONS]
        + [f"mass_{a}_to_{b}" for a, b in TRANSITIONS]
    )
    save_csv(
        out / "error_profiles.csv",
        err_header,
        [
            [i, int(abs_crossline[i]), overall_err[i]]
            + [cls_err[c][i] for c in CLASSES]
            + [trans_rate[pair][i] for pair in TRANSITIONS]
            + [trans_mass[pair][i] for pair in TRANSITIONS]
            for i in range(z_n)
        ],
    )

    series = [("overall", overall_err, {"color": "black", "lw": 2.0})]
    for c in CLASSES:
        series.append((f"true {c} error", cls_err[c], {"color": colors[c], "lw": 1.5}))
    plot_lines(
        fig_dir / "error_profiles.png",
        z,
        series,
        "test crossline z",
        "error rate",
        "Per-crossline error (overall and true classes 3–6)",
        ylim=(0, 1),
    )

    series = []
    tcolors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (pair, col) in zip(TRANSITIONS, tcolors):
        series.append((f"{pair[0]}→{pair[1]}", trans_rate[pair], {"color": col, "lw": 1.6}))
    plot_lines(
        fig_dir / "transition_rates.png",
        z,
        series,
        "test crossline z",
        "P(pred=b | true=a)",
        "Main confusion transitions along the test volume",
        ylim=(0, 1),
    )

    # ------------------------------------------------------------------
    # 4. IoU by crossline
    # ------------------------------------------------------------------
    iou_z = {c: np.full(z_n, np.nan) for c in CLASSES}
    for i in range(z_n):
        for c in CLASSES:
            iou_z[c][i] = iou_binary(pred[i], gt[i], c)

    half = z_n // 2
    iou_first = {c: nanmean(iou_z[c][:half]) for c in CLASSES}
    iou_last = {c: nanmean(iou_z[c][half:]) for c in CLASSES}

    save_csv(
        out / "iou_by_crossline.csv",
        ["z", "abs_crossline"] + [f"iou_{c}" for c in CLASSES],
        [
            [i, int(abs_crossline[i])] + [iou_z[c][i] for c in CLASSES]
            for i in range(z_n)
        ],
    )
    series = [(f"IoU {c}", iou_z[c], {"color": colors[c], "lw": 1.6}) for c in CLASSES]
    plot_lines(
        fig_dir / "iou_by_crossline.png",
        z,
        series,
        "test crossline z",
        "IoU",
        "Per-crossline IoU of classes 3–6 (NaN skipped where union=0)",
        ylim=(0, 1),
    )

    # ------------------------------------------------------------------
    # 5. Dependency on prediction_count
    # ------------------------------------------------------------------
    overlap_rows = []
    overlap_summary = []
    wrong = pred != gt
    for name, lo, hi in OVERLAP_BINS:
        m = (count >= lo) & (count <= hi)
        n = int(m.sum())
        row = {
            "bin": name,
            "count_min": lo,
            "count_max": hi if hi < 10_000 else int(count.max()),
            "n_voxels": n,
            "fraction_of_volume": float(n / count.size) if count.size else 0.0,
            "overall_error": float(wrong[m].mean()) if n else float("nan"),
        }
        for c in CLASSES:
            true_c = m & (gt == c)
            nc = int(true_c.sum())
            row[f"n_true_{c}"] = nc
            row[f"error_true_{c}"] = (
                float((pred[true_c] != c).mean()) if nc else float("nan")
            )
            row[f"iou_{c}"] = iou_binary(pred[m], gt[m], c) if n else float("nan")
        overlap_summary.append(row)
        overlap_rows.append(
            [row["bin"], row["n_voxels"], row["fraction_of_volume"], row["overall_error"]]
            + [row[f"error_true_{c}"] for c in CLASSES]
            + [row[f"iou_{c}"] for c in CLASSES]
            + [row[f"n_true_{c}"] for c in CLASSES]
        )

    save_csv(
        out / "overlap_error_analysis.csv",
        ["bin", "n_voxels", "fraction_of_volume", "overall_error"]
        + [f"error_true_{c}" for c in CLASSES]
        + [f"iou_{c}" for c in CLASSES]
        + [f"n_true_{c}" for c in CLASSES],
        overlap_rows,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = [r["bin"] for r in overlap_summary]
    axes[0].bar(bins, [r["overall_error"] for r in overlap_summary], color="#4c72b0")
    style_axes(axes[0], "prediction_count bin", "overall error", "Error vs overlap count")
    axes[0].set_ylim(0, 1)
    width = 0.18
    x = np.arange(len(bins))
    for i, c in enumerate(CLASSES):
        axes[1].bar(
            x + (i - 1.5) * width,
            [r[f"iou_{c}"] for r in overlap_summary],
            width=width,
            color=colors[c],
            label=f"IoU {c}",
        )
    axes[1].set_xticks(x, bins)
    style_axes(axes[1], "prediction_count bin", "IoU", "Class 3–6 IoU vs overlap count")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "overlap_error.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Test-beginning vs test-end (quartiles)
    # ------------------------------------------------------------------
    q_rows = []
    q_summary = []
    for label, sl in quartile_slices(z_n):
        g = gt[sl]
        p = pred[sl]
        rec = {
            "quartile": label,
            "z_start": int(sl.start),
            "z_end": int(sl.stop) - 1,
            "abs_crossline_start": int(abs_crossline[sl.start]),
            "abs_crossline_end": int(abs_crossline[sl.stop - 1]),
            "n_voxels": int(g.size),
            "overall_error": float((p != g).mean()),
        }
        for c in CLASSES:
            rec[f"gt_frac_{c}"] = float((g == c).mean())
            rec[f"pred_frac_{c}"] = float((p == c).mean())
            rec[f"iou_{c}"] = iou_binary(p, g, c)
            true_c = g == c
            nc = int(true_c.sum())
            rec[f"error_true_{c}"] = (
                float((p[true_c] != c).mean()) if nc else float("nan")
            )
        for a, b in TRANSITIONS:
            true_a = g == a
            n = int(true_a.sum())
            rec[f"rate_{a}_to_{b}"] = (
                float((p[true_a] == b).mean()) if n else float("nan")
            )
        q_summary.append(rec)
        q_rows.append(
            [
                rec["quartile"],
                rec["z_start"],
                rec["z_end"],
                rec["abs_crossline_start"],
                rec["abs_crossline_end"],
                rec["n_voxels"],
                rec["overall_error"],
            ]
            + [rec[f"gt_frac_{c}"] for c in CLASSES]
            + [rec[f"pred_frac_{c}"] for c in CLASSES]
            + [rec[f"iou_{c}"] for c in CLASSES]
            + [rec[f"error_true_{c}"] for c in CLASSES]
            + [rec[f"rate_{a}_to_{b}"] for a, b in TRANSITIONS]
        )

    save_csv(
        out / "quartile_comparison.csv",
        [
            "quartile",
            "z_start",
            "z_end",
            "abs_crossline_start",
            "abs_crossline_end",
            "n_voxels",
            "overall_error",
        ]
        + [f"gt_frac_{c}" for c in CLASSES]
        + [f"pred_frac_{c}" for c in CLASSES]
        + [f"iou_{c}" for c in CLASSES]
        + [f"error_true_{c}" for c in CLASSES]
        + [f"rate_{a}_to_{b}" for a, b in TRANSITIONS],
        q_rows,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    qn = [r["quartile"] for r in q_summary]
    x = np.arange(len(qn))
    w = 0.18
    for i, c in enumerate(CLASSES):
        axes[0].bar(x + (i - 1.5) * w, [r[f"gt_frac_{c}"] for r in q_summary], w, color=colors[c], label=f"GT {c}")
        axes[1].bar(x + (i - 1.5) * w, [r[f"pred_frac_{c}"] for r in q_summary], w, color=colors[c], label=f"Pred {c}")
        axes[2].bar(x + (i - 1.5) * w, [r[f"iou_{c}"] for r in q_summary], w, color=colors[c], label=f"IoU {c}")
    for ax, title, ylab, ylim in zip(
        axes,
        ["GT class fraction", "Predicted class fraction", "IoU"],
        ["fraction", "fraction", "IoU"],
        [(0, None), (0, None), (0, 1)],
    ):
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        style_axes(ax, "test quartile (Q1=near val, Q4=farthest)", ylab, title)
        if ylim[1] is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "quartile_comparison.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Extra: a few GT / Pred / error slices for visual inspection
    # ------------------------------------------------------------------
    vis_z = [0, z_n // 4, z_n // 2, (3 * z_n) // 4, z_n - 1]
    cmap = plt.get_cmap("tab10", 10)
    for zi in vis_z:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
        axes[0].imshow(gt[zi], cmap=cmap, vmin=0, vmax=9, interpolation="nearest", aspect="auto")
        axes[0].set_title(f"GT z={zi} (xl={int(abs_crossline[zi])})")
        axes[1].imshow(pred[zi], cmap=cmap, vmin=0, vmax=9, interpolation="nearest", aspect="auto")
        axes[1].set_title(f"Pred z={zi}")
        err_img = np.where(pred[zi] != gt[zi], gt[zi], np.nan)
        axes[2].imshow(gt[zi], cmap="gray", vmin=0, vmax=9, interpolation="nearest", aspect="auto", alpha=0.25)
        axes[2].imshow(err_img, cmap=cmap, vmin=0, vmax=9, interpolation="nearest", aspect="auto")
        axes[2].set_title("Errors (colored by true class)")
        for ax in axes:
            ax.set_xlabel("inline X")
            ax.set_ylabel("depth Y")
        fig.tight_layout()
        fig.savefig(fig_dir / f"slice_z{zi:03d}.png", dpi=120)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Trend numbers
    # ------------------------------------------------------------------
    def slope(xs, ys):
        m = np.isfinite(ys)
        if m.sum() < 3:
            return float("nan")
        return float(np.polyfit(xs[m], ys[m], 1)[0])

    gt6_slope = slope(z.astype(float), gt_frac[6])
    pr6_slope = slope(z.astype(float), pr_frac[6])
    gt4_slope = slope(z.astype(float), gt_frac[4])
    pr4_slope = slope(z.astype(float), pr_frac[4])
    err6_slope = slope(z.astype(float), cls_err[6])
    trans65_slope = slope(z.astype(float), trans_rate[(6, 5)])
    depth6_gt_slope = slope(z.astype(float), depth_store[6]["gt_mean"])
    depth6_pr_slope = slope(z.astype(float), depth_store[6]["pr_mean"])

    # overlap: compare low-count vs interior
    low = next(r for r in overlap_summary if r["bin"] == "1-3")
    interior = next(r for r in overlap_summary if r["bin"] == "8")
    high = next(r for r in overlap_summary if r["bin"] == ">=12")

    q1 = q_summary[0]
    q4 = q_summary[-1]

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])

    summary = {
        "volume_shape": [int(z_n), int(y_n), int(x_n)],
        "survey_test_split": [809, 951],
        "n_voxels": int(pred.size),
        "overall_error": float((pred != gt).mean()),
        "official_volume_miou_no_bg": 0.5884245804616746,
        "crossline": {
            "iou_first_half": iou_first,
            "iou_second_half": iou_last,
            "iou_delta_last_minus_first": {c: iou_last[c] - iou_first[c] for c in CLASSES},
            "gt_frac_slope_per_crossline": {3: slope(z.astype(float), gt_frac[3]), 4: gt4_slope, 5: slope(z.astype(float), gt_frac[5]), 6: gt6_slope},
            "pred_frac_slope_per_crossline": {3: slope(z.astype(float), pr_frac[3]), 4: pr4_slope, 5: slope(z.astype(float), pr_frac[5]), 6: pr6_slope},
            "class6_gt_vs_pred_corr": corr(gt_frac[6], pr_frac[6]),
            "class4_gt_vs_pred_corr": corr(gt_frac[4], pr_frac[4]),
            "true6_error_slope": err6_slope,
            "rate_6_to_5_slope": trans65_slope,
        },
        "depth": {
            "class6_gt_mean_y_slope": depth6_gt_slope,
            "class6_pred_mean_y_slope": depth6_pr_slope,
            "class6_gt_mean_y_first_half": nanmean(depth_store[6]["gt_mean"][:half]),
            "class6_gt_mean_y_second_half": nanmean(depth_store[6]["gt_mean"][half:]),
            "class6_pred_mean_y_first_half": nanmean(depth_store[6]["pr_mean"][:half]),
            "class6_pred_mean_y_second_half": nanmean(depth_store[6]["pr_mean"][half:]),
            "class4_gt_mean_y_first_half": nanmean(depth_store[4]["gt_mean"][:half]),
            "class4_gt_mean_y_second_half": nanmean(depth_store[4]["gt_mean"][half:]),
            "class5_gt_mean_y_first_half": nanmean(depth_store[5]["gt_mean"][:half]),
            "class5_gt_mean_y_second_half": nanmean(depth_store[5]["gt_mean"][half:]),
        },
        "overlap": overlap_summary,
        "overlap_conclusion": {
            "low_count_1_3_error": low["overall_error"],
            "interior_count_8_error": interior["overall_error"],
            "high_count_ge12_error": high["overall_error"],
            "low_minus_interior": low["overall_error"] - interior["overall_error"],
            "low_count_volume_fraction": low["fraction_of_volume"],
        },
        "quartiles": q_summary,
        "quartile_shift": {
            "gt_frac_6_Q1": q1["gt_frac_6"],
            "gt_frac_6_Q4": q4["gt_frac_6"],
            "pred_frac_6_Q1": q1["pred_frac_6"],
            "pred_frac_6_Q4": q4["pred_frac_6"],
            "gt_frac_4_Q1": q1["gt_frac_4"],
            "gt_frac_4_Q4": q4["gt_frac_4"],
            "iou_4_Q1": q1["iou_4"],
            "iou_4_Q4": q4["iou_4"],
            "iou_6_Q1": q1["iou_6"],
            "iou_6_Q4": q4["iou_6"],
            "rate_6_to_5_Q1": q1["rate_6_to_5"],
            "rate_6_to_5_Q4": q4["rate_6_to_5"],
            "rate_4_to_3_Q1": q1["rate_4_to_3"],
            "rate_4_to_3_Q4": q4["rate_4_to_3"],
            "overall_error_Q1": q1["overall_error"],
            "overall_error_Q4": q4["overall_error"],
        },
    }

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if np.isnan(v) else v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    (out / "summary.json").write_text(
        json.dumps(_jsonify(summary), indent=2), encoding="utf-8"
    )

    # README is filled after we know the numbers — written by a helper below
    write_readme(out, summary, iou_first, iou_last, q1, q4, low, interior, high)

    print(f"Wrote diagnostics to {out}")
    print("overall_error", summary["overall_error"])
    print("iou first/last", iou_first, iou_last)
    print("Q1 vs Q4 error", q1["overall_error"], q4["overall_error"])
    print("overlap low vs interior", low["overall_error"], interior["overall_error"])


def write_readme(out: Path, summary, iou_first, iou_last, q1, q4, low, interior, high):
    qs = summary["quartile_shift"]
    ov = summary["overlap_conclusion"]
    cr = summary["crossline"]
    dp = summary["depth"]

    def f(v, nd=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
        return f"{v:.{nd}f}"

    md = f"""# Baseline diagnostics (volume-level, no retraining)

Input volumes: `source/test_logs_volume/{{prediction_volume,ground_truth_volume,prediction_count}}.npy`

Volume shape: `{summary['volume_shape']}` (test split survey crosslines 809–950).
Official baseline mIoU without background: **0.58842**. Pixel error on the reconstructed volume: **{f(summary['overall_error'])}**.

This folder is produced by `source/diagnose_baseline.py` (CPU-only). The model, checkpoint, loss, split, patch and stride were not changed.

## Files

| file | contents |
|---|---|
| `crossline_profiles.csv` | GT vs predicted occupancy of classes 3–6 on every test crossline |
| `depth_profiles.csv` | mean / median / std depth (Y) of classes 4–6 |
| `error_profiles.csv` | overall and per-class error, plus 3→2 / 4→3 / 5→4 / 6→5 / 6→8 |
| `iou_by_crossline.csv` | IoU of classes 3–6 on each crossline (blank if union=0) |
| `overlap_error_analysis.csv` | error and IoU stratified by `prediction_count` |
| `quartile_comparison.csv` | first / second / third / last 25% of the test volume |
| `summary.json` | machine-readable headline numbers |
| `figures/` | PNG plots and five GT/Pred/error slices |

## Headline numbers

### IoU 3–6, first half vs second half of test

| class | first half | second half | delta |
|---|---|---|---|
| 3 | {f(iou_first[3])} | {f(iou_last[3])} | {f(iou_last[3]-iou_first[3])} |
| 4 | {f(iou_first[4])} | {f(iou_last[4])} | {f(iou_last[4]-iou_first[4])} |
| 5 | {f(iou_first[5])} | {f(iou_last[5])} | {f(iou_last[5]-iou_first[5])} |
| 6 | {f(iou_first[6])} | {f(iou_last[6])} | {f(iou_last[6]-iou_first[6])} |

### Quartiles (Q1 = near val, Q4 = farthest from train)

| | Q1 | Q4 |
|---|---|---|
| overall error | {f(qs['overall_error_Q1'])} | {f(qs['overall_error_Q4'])} |
| GT frac 4 | {f(qs['gt_frac_4_Q1'])} | {f(qs['gt_frac_4_Q4'])} |
| Pred frac 4 | {f(q1['pred_frac_4'])} | {f(q4['pred_frac_4'])} |
| GT frac 6 | {f(qs['gt_frac_6_Q1'])} | {f(qs['gt_frac_6_Q4'])} |
| Pred frac 6 | {f(qs['pred_frac_6_Q1'])} | {f(qs['pred_frac_6_Q4'])} |
| IoU 4 | {f(qs['iou_4_Q1'])} | {f(qs['iou_4_Q4'])} |
| IoU 6 | {f(qs['iou_6_Q1'])} | {f(qs['iou_6_Q4'])} |
| P(pred=3 \\| true=4) | {f(qs['rate_4_to_3_Q1'])} | {f(qs['rate_4_to_3_Q4'])} |
| P(pred=5 \\| true=6) | {f(qs['rate_6_to_5_Q1'])} | {f(qs['rate_6_to_5_Q4'])} |

### Overlap (`prediction_count`)

| bin | volume fraction | overall error |
|---|---|---|
| count 1–3 | {f(ov['low_count_volume_fraction'])} | {f(ov['low_count_1_3_error'])} |
| count 8 (interior) | {f(interior['fraction_of_volume'])} | {f(ov['interior_count_8_error'])} |
| count ≥12 | {f(high['fraction_of_volume'])} | {f(ov['high_count_ge12_error'])} |

Low-count minus interior error: **{f(ov['low_minus_interior'])}**. Count 1–3 is only {f(100*ov['low_count_volume_fraction'], 2)}% of the volume.

Class-6 mean depth Y: GT first half {f(dp['class6_gt_mean_y_first_half'], 1)} → second half {f(dp['class6_gt_mean_y_second_half'], 1)}; Pred first half {f(dp['class6_pred_mean_y_first_half'], 1)} → second half {f(dp['class6_pred_mean_y_second_half'], 1)}.

GT class-6 occupancy slope vs z: {f(cr['gt_frac_slope_per_crossline'][6], 6)} per crossline.
Pred class-6 occupancy slope vs z: {f(cr['pred_frac_slope_per_crossline'][6], 6)} per crossline.
Corr(GT frac 6, Pred frac 6) = {f(cr['class6_gt_vs_pred_corr'])}.
Corr(GT frac 4, Pred frac 4) = {f(cr['class4_gt_vs_pred_corr'])}.

---

A. Spatial shift evidence

B. Class 3–6 behavior

C. Patch-boundary / overlap evidence

D. Most likely explanation

E. What is NOT supported by the data

F. Recommended next experiment
"""
    # The A–F block is completed after the run, from the actual numbers, in a
    # second pass below so the narrative cannot drift from the CSV.
    (out / "README.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
