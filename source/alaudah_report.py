"""Create a multi-page HTML report for 3D U-Net evaluation on Alaudah.

Creates:
  test_result/index.html       overview + pooled test1/test2 metrics
  test_result/test1.html       test1 metrics, confusion matrix, patch visualizations
  test_result/test2.html       test2 metrics, confusion matrix, patch visualizations
  test_result/combined.html    pooled test1 + test2 metrics
  test_result/training.html    train/validation loss + class distribution

Pooled metrics are recalculated by summing confusion-matrix COUNTS and then
calculating metrics from the summed matrix. They are not the arithmetic mean
of metrics from test1 and test2.

Existing test directories and visualizations are read-only; they are not changed.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_CLASSES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a multi-page Alaudah 3D U-Net HTML report")
    parser.add_argument("--test1_dir", type=Path, default=Path("./test_logs/alaudah/test1"))
    parser.add_argument("--test2_dir", type=Path, default=Path("./test_logs/alaudah/test2"))
    parser.add_argument("--combined_dir", type=Path, default=Path("./test_logs/alaudah/combined"),
                        help="Folder where pooled combined CSV files will be saved")
    parser.add_argument("--train_loss_csv", type=Path, default=Path("./train_logs/alaudah/train_loss.csv"))
    parser.add_argument("--val_loss_csv", type=Path, default=Path("./train_logs/alaudah/val_loss.csv"))
    parser.add_argument("--train_h5", type=Path, default=Path("../pytorch-3dunet-master/my_data/benchmark/train.h5"))
    parser.add_argument("--val_h5", type=Path, default=Path("../pytorch-3dunet-master/my_data/benchmark/val.h5"))
    parser.add_argument("--test1_h5", type=Path, default=Path("../pytorch-3dunet-master/my_data/benchmark/test1.h5"))
    parser.add_argument("--test2_h5", type=Path, default=Path("../pytorch-3dunet-master/my_data/benchmark/test2.h5"))
    parser.add_argument("--output_dir", type=Path, default=Path("./test_logs/alaudah/test_result"))
    parser.add_argument("--max_visual_patches", type=int, default=16)
    return parser.parse_args()


def data_uri_from_bytes(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def fig_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    return data_uri_from_bytes(buffer.getvalue())


def image_from_path(path: Path, alt: str, css: str = "report-image") -> str:
    if not path.exists():
        return '<p class="muted">Файл отсутствует; блок пропущен.</p>'
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    uri = data_uri_from_bytes(path.read_bytes(), mime)
    return f'<img class="{css}" alt="{html.escape(alt)}" src="{uri}">'


def confusion_from_arrays(gt_path: Path, pred_path: Path, n_classes: int = N_CLASSES) -> np.ndarray | None:
    if not gt_path.is_file() or not pred_path.is_file():
        return None
    gt = np.load(gt_path, mmap_mode="r").reshape(-1)
    pred = np.load(pred_path, mmap_mode="r").reshape(-1)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt_path} has {gt.shape}, {pred_path} has {pred.shape}")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    chunk = 8_000_000
    for start in range(0, gt.size, chunk):
        y = np.asarray(gt[start:start + chunk], dtype=np.int64)
        p = np.asarray(pred[start:start + chunk], dtype=np.int64)
        valid = (y >= 0) & (y < n_classes) & (p >= 0) & (p < n_classes)
        if not np.all(valid):
            raise ValueError(f"Invalid class id outside 0..{n_classes - 1} in {gt_path.parent}")
        cm += np.bincount(n_classes * y + p, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    return cm


def read_matrix_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        matrix = np.loadtxt(path, delimiter=",")
        if matrix.shape == (N_CLASSES, N_CLASSES) and np.isfinite(matrix).all():
            return matrix
    except Exception:
        pass
    try:
        df = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce")
        arr = df.to_numpy(dtype=float)
        arr = arr[~np.all(np.isnan(arr), axis=1)]
        arr = arr[:, ~np.all(np.isnan(arr), axis=0)]
        if arr.shape == (N_CLASSES, N_CLASSES) and np.isfinite(arr).all():
            return arr
    except Exception as exc:
        print(f"Warning: cannot read matrix {path}: {exc}")
    return None


def support_from_class_csv(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        if "class" not in df.columns:
            return None
        result = np.zeros(N_CLASSES, dtype=np.int64)
        found = set()
        for _, row in df.iterrows():
            try:
                c = int(row["class"])
            except (TypeError, ValueError):
                continue
            if 0 <= c < N_CLASSES:
                value = row.get("pixels", row.get("support", row.get("pixels_support", np.nan)))
                if pd.notna(value) and str(value).strip() != "":
                    result[c] = int(float(value))
                    found.add(c)
        return result if len(found) == N_CLASSES else None
    except Exception:
        return None


def load_confusion(folder: Path) -> tuple[np.ndarray, str]:
    gt_path = folder / "ground_truth_volume.npy"
    pred_path = folder / "prediction_volume.npy"
    cm = confusion_from_arrays(gt_path, pred_path)
    if cm is not None:
        return cm, "Точно пересчитана из ground_truth_volume.npy и prediction_volume.npy"

    for name in ("confusion_matrix_counts.csv", "combined_confusion_matrix_counts.csv"):
        cm = read_matrix_csv(folder / name)
        if cm is not None:
            if not np.allclose(cm, np.rint(cm)) or (cm < 0).any():
                raise ValueError(f"{folder / name} должен содержать неотрицательные целые количества")
            return np.rint(cm).astype(np.int64), f"Прочитана из {name} (counts)"

    percent = read_matrix_csv(folder / "confusion_matrix_percent.csv")
    if percent is None:
        percent = read_matrix_csv(folder / "combined_confusion_matrix_percent.csv")
    support = support_from_class_csv(folder / "class_metrics.csv")
    if support is None:
        support = support_from_class_csv(folder / "combined_class_metrics.csv")
    if percent is not None and support is not None:
        approx = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for c in range(N_CLASSES):
            row = np.rint(percent[c] / 100 * support[c]).astype(np.int64)
            row[int(np.argmax(percent[c]))] += int(support[c] - row.sum())
            if (row < 0).any():
                raise ValueError(f"Cannot reconstruct row {c} from {folder / 'confusion_matrix_percent.csv'}")
            approx[c] = row
        return approx, "ПРИБЛИЖЁННО восстановлена из округлённой процентной матрицы и support"

    raise FileNotFoundError(
        f"Нет достаточных файлов результатов в {folder}. Нужны prediction_volume.npy + "
        "ground_truth_volume.npy (предпочтительно), либо confusion_matrix_counts.csv, "
        "либо confusion_matrix_percent.csv + class_metrics.csv."
    )


def metrics_from_cm(cm: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    cm_f = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm_f)
    support = cm_f.sum(axis=1)
    predicted = cm_f.sum(axis=0)
    total = float(cm_f.sum())
    union = support + predicted - tp
    precision = np.divide(tp, predicted, out=np.full(N_CLASSES, np.nan), where=predicted > 0)
    recall = np.divide(tp, support, out=np.full(N_CLASSES, np.nan), where=support > 0)
    iou = np.divide(tp, union, out=np.full(N_CLASSES, np.nan), where=union > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.full(N_CLASSES, np.nan), where=(precision + recall) > 0)
    frequency = support / total if total else np.zeros(N_CLASSES, dtype=float)
    fwiou_contribution = frequency * np.nan_to_num(iou, nan=0.0)
    summary = {
        "pixel_accuracy": float(tp.sum() / total) if total else float("nan"),
        "mean_class_accuracy": float(np.nanmean(recall)) if np.isfinite(recall).any() else float("nan"),
        "mean_iou_all_classes": float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan"),
        "mean_iou_classes_1_5": float(np.nanmean(iou[1:])) if np.isfinite(iou[1:]).any() else float("nan"),
        "frequency_weighted_iou": float(fwiou_contribution.sum()),
        "evaluated_voxels": total,
    }
    class_df = pd.DataFrame({
        "Класс": np.arange(N_CLASSES),
        "Support (воксели)": support.astype(np.int64),
        "Частота": frequency,
        "Accuracy / Recall": recall,
        "Precision": precision,
        "IoU": iou,
        "F1": f1,
        "FWIoU": fwiou_contribution,
    })
    return summary, class_df


def add_mean_row(class_df: pd.DataFrame, summary: dict[str, float]) -> pd.DataFrame:
    # Class-wise scores use macro averages; frequency is a distribution and therefore sums to 100%.
    rate_cols = ["Accuracy / Recall", "Precision", "IoU", "F1"]
    row: dict[str, Any] = {"Класс": "Среднее / итог"}
    row["Support (воксели)"] = int(class_df["Support (воксели)"].sum())
    row["Частота"] = float(class_df["Частота"].sum())
    for col in rate_cols:
        vals = pd.to_numeric(class_df[col], errors="coerce")
        row[col] = float(vals.mean(skipna=True)) if vals.notna().any() else np.nan
    row["FWIoU"] = float(class_df["FWIoU"].sum())
    # Also store summary values in table context for transparent totals.
    return pd.concat([class_df, pd.DataFrame([row])], ignore_index=True)


def format_class_table(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for col in ("Частота",):
        formatted[col] = formatted[col].map(lambda x: "—" if pd.isna(x) else f"{x * 100:.2f}%")
    for col in ("Accuracy / Recall", "Precision", "IoU", "F1", "FWIoU"):
        formatted[col] = formatted[col].map(lambda x: "—" if pd.isna(x) else f"{x:.4f}")
    formatted["Support (воксели)"] = formatted["Support (воксели)"].map(lambda x: "—" if pd.isna(x) else f"{int(x):,}")
    return formatted.to_html(index=False, classes="data-table", border=0, escape=True)


def summary_cards(summary: dict[str, float]) -> str:
    descriptions = {
        "pixel_accuracy": ("Pixel Accuracy", "Доля всех правильно классифицированных вокселей"),
        "mean_class_accuracy": ("Mean Class Accuracy", "Средний recall по шести классам"),
        "mean_iou_all_classes": ("Mean IoU — все 6 классов", "Макро-среднее IoU по всем реальным фациям"),
        "frequency_weighted_iou": ("Frequency Weighted IoU", "IoU, взвешенный частотой истинных классов"),
    }
    items = []
    for key in descriptions:
        if key not in summary:
            continue
        label, desc = descriptions[key]
        value = summary[key]
        shown = "—" if not np.isfinite(value) else f"{value * 100:.2f}%"
        items.append(f'<div class="metric"><div class="metric-name">{html.escape(label)}</div><div class="metric-value">{shown}</div><div class="metric-desc">{html.escape(desc)}</div></div>')
    if "evaluated_voxels" in summary:
        items.append(f'<div class="metric"><div class="metric-name">Оценено вокселей</div><div class="metric-value">{summary["evaluated_voxels"]:,.0f}</div><div class="metric-desc">Количество вокселей в матрице ошибок</div></div>')
    return "".join(items)


def plot_confusion(cm: np.ndarray, title: str) -> str:
    row_sums = cm.sum(axis=1, keepdims=True)
    percent = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    im = ax.imshow(percent, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% внутри истинного класса")
    ax.set_xticks(range(N_CLASSES), labels=range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES), labels=range(N_CLASSES))
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = percent[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color="white" if val > 52 else "#17202a", fontsize=8)
    fig.tight_layout()
    return fig_data_uri(fig)


def read_loss(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        # Search only for a unique file with the same basename under train_logs.
        root = Path("train_logs")
        matches = list(root.rglob(path.name)) if root.exists() else []
        if len(matches) == 1:
            path = matches[0]
            print(f"Found loss CSV: {path}")
        else:
            print(f"Warning: loss CSV not found: {path}")
            return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        xcol = next((c for c in ("Step", "step", "Epoch", "epoch") if c in df.columns), None)
        ycol = next((c for c in ("Value", "value", "Loss", "loss") if c in df.columns), None)
        if xcol is None or ycol is None:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric) < 2:
                print(f"Warning: cannot identify loss columns in {path}")
                return None
            xcol, ycol = numeric[:2]
        return df[xcol].to_numpy(), df[ycol].to_numpy()
    except Exception as exc:
        print(f"Warning: cannot read {path}: {exc}")
        return None


def loss_plot(train_path: Path, val_path: Path) -> str | None:
    train, val = read_loss(train_path), read_loss(val_path)
    if train is None and val is None:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    if train is not None:
        ax.plot(train[0], train[1], label="Train loss", linewidth=2.2)
    if val is not None:
        ax.plot(val[0], val[1], label="Validation loss", linewidth=2.2)
    ax.set_xlabel("Step / epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.grid(True, alpha=0.23)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig_data_uri(fig)


def count_labels_h5(path: Path) -> np.ndarray | None:
    if not path.is_file():
        print(f"Warning: class-distribution HDF5 not found: {path}")
        return None
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    try:
        with h5py.File(path, "r") as f:
            if "label" not in f:
                print(f"Warning: no 'label' dataset in {path}")
                return None
            ds = f["label"]
            step = max(1, min(16, ds.shape[0]))
            for start in range(0, ds.shape[0], step):
                arr = np.asarray(ds[start:start + step]).reshape(-1).astype(np.int64, copy=False)
                valid = arr[(arr >= 0) & (arr < N_CLASSES)]
                counts += np.bincount(valid, minlength=N_CLASSES)[:N_CLASSES]
        return counts
    except Exception as exc:
        print(f"Warning: cannot count labels in {path}: {exc}")
        return None


def count_labels_npy(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.load(path, mmap_mode="r").reshape(-1)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for start in range(0, arr.size, 8_000_000):
        part = np.asarray(arr[start:start + 8_000_000]).astype(np.int64, copy=False)
        valid = part[(part >= 0) & (part < N_CLASSES)]
        counts += np.bincount(valid, minlength=N_CLASSES)[:N_CLASSES]
    return counts


def percentages(counts: np.ndarray | None) -> np.ndarray | None:
    if counts is None:
        return None
    return counts * (100.0 / counts.sum()) if counts.sum() else np.zeros_like(counts, dtype=float)


def class_distribution(args, output_csv: Path) -> tuple[pd.DataFrame | None, str | None]:
    train = count_labels_h5(args.train_h5)
    val = count_labels_h5(args.val_h5)
    t1 = count_labels_npy(args.test1_dir / "ground_truth_volume.npy")
    if t1 is None:
        t1 = count_labels_h5(args.test1_h5)
    t2 = count_labels_npy(args.test2_dir / "ground_truth_volume.npy")
    if t2 is None:
        t2 = count_labels_h5(args.test2_h5)
    if any(x is None for x in (train, val, t1, t2)):
        print("Warning: cannot create full class distribution; at least one split's labels are unavailable.")
        return None, None
    test = t1 + t2
    df = pd.DataFrame({
        "Класс": np.arange(N_CLASSES),
        "Train, %": percentages(train),
        "Validation, %": percentages(val),
        "Test1, %": percentages(t1),
        "Test2, %": percentages(t2),
        "Combined test, %": percentages(test),
        "Train voxels": train,
        "Validation voxels": val,
        "Test1 voxels": t1,
        "Test2 voxels": t2,
        "Combined test voxels": test,
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"Class distribution saved: {output_csv.resolve()}")
    return df, str(output_csv)


def distribution_plot(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(N_CLASSES)
    width = 0.19
    for offset, col, label in [
        (-1.5 * width, "TRAIN", "Train"),
        (-0.5 * width, "VAL", "Validation"),
        (0.5 * width, "TEST1", "Test1"),
        (1.5 * width, "TEST2", "Test2"),
    ]:
        ax.bar(x + offset, df[col], width, label=label)
    ax.set_xticks(x, labels=df["Class"].astype(str))
    ax.set_xlabel("Class")
    ax.set_ylabel("Share of voxels (%)")
    ax.set_title("Class distribution by split")
    ax.grid(axis="y", alpha=0.23)
    ax.legend(frameon=False, ncols=4)
    fig.tight_layout()
    return fig_data_uri(fig)


def render_class_distribution(df: pd.DataFrame | None) -> str:
    if df is None:
        return '<p class="muted">Распределение классов не рассчитано: проверьте пути train/val/test HDF5 и сохранённые ground_truth_volume.npy.</p>'
    shown = df.copy()
    pct_cols = [c for c in ("TRAIN", "VAL", "TEST1", "TEST2", "TEST") if c in shown.columns]
    for col in pct_cols:
        shown[col] = shown[col].map(lambda x: f"{x:.2f}%")
    for col in shown.columns:
        if "voxels" in col.lower():
            shown[col] = shown[col].map(lambda x: f"{int(x):,}")
    return shown.to_html(index=False, classes="data-table", border=0, escape=True)


def visualizations_html(folder: Path, max_patches: int) -> str:
    vis_dir = folder / "visualizations"
    if not vis_dir.is_dir():
        return '<p class="muted">Папка visualizations отсутствует.</p>'
    patches = sorted((p for p in vis_dir.iterdir() if p.is_dir() and p.name.startswith("patch_")), key=lambda p: p.name)
    if not patches:
        return '<p class="muted">В папке visualizations нет patch_XXX.</p>'
    cells = []
    image_files = [("seismic.png", "Сейсмика"), ("ground_truth.png", "Истинная разметка"), ("prediction.png", "Предсказание")]
    for patch in patches[:max_patches]:
        group = []
        for filename, label in image_files:
            path = patch / filename
            group.append(f'<div class="visual-card"><div class="visual-title">{html.escape(label)}</div>{image_from_path(path, label, "patch-image")}</div>')
        cells.append(f'<article class="patch"><h3>{html.escape(patch.name)}</h3><div class="patch-grid">{"".join(group)}</div></article>')
    if len(patches) > max_patches:
        cells.append(f'<p class="muted">Показаны первые {max_patches} из {len(patches)} патчей. Исходные изображения не изменены.</p>')
    return "\n".join(cells)


def summary_comparison_table(summaries: dict[str, dict[str, float]]) -> str:
    rows = []
    labels = [
        ("Pixel Accuracy", "pixel_accuracy"),
        ("Mean Class Accuracy", "mean_class_accuracy"),
        ("Mean IoU — все 6 классов", "mean_iou_all_classes"),
        ("Mean IoU — классы 1–5", "mean_iou_classes_1_5"),
        ("Frequency Weighted IoU", "frequency_weighted_iou"),
        ("Количество вокселей", "evaluated_voxels"),
    ]
    for label, key in labels:
        row: dict[str, Any] = {"Метрика": label}
        for split in ("test1", "test2", "combined"):
            value = summaries.get(split, {}).get(key, np.nan)
            if not np.isfinite(value):
                text = "—"
            elif key == "evaluated_voxels":
                text = f"{value:,.0f}"
            else:
                text = f"{value * 100:.2f}%"
            row[split] = text
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={"test1": "Test 1", "test2": "Test 2", "combined": "Объединённый тест"}).to_html(index=False, classes="data-table", border=0, escape=True)


def page_shell(title: str, active: str, content: str) -> str:
    nav = [
        ("index", "Обзор", "index.html"),
        ("combined", "Итоговый результат", "combined.html"),
        ("test1", "Test 1", "test1.html"),
        ("test2", "Test 2", "test2.html"),
        ("training", "Обучение и данные", "training.html"),
    ]
    links = "".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{label}</a>'
        for key, label, href in nav
    )
    return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#657383; --line:#e0e6ed; --panel:#fff; --bg:#f2f5f9; --nav:#172b4d; --accent:#176b87; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.5; }}
.topbar {{ background:var(--nav); color:#fff; padding:22px max(22px, calc((100vw - 1420px)/2)); }}
.brand {{ font-size:13px; text-transform:uppercase; letter-spacing:.12em; color:#c6d5e5; }} .topbar h1 {{ margin:5px 0 0; font-size:clamp(24px,3vw,34px); }}
.nav {{ display:flex; gap:7px; flex-wrap:wrap; padding:12px max(22px, calc((100vw - 1420px)/2)); background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:4; }}
.nav-link {{ text-decoration:none; padding:9px 13px; border-radius:8px; color:#34465a; font-weight:600; font-size:14px; }} .nav-link:hover {{ background:#f0f5f8; }} .nav-link.active {{ background:#dceff5; color:#07556d; }}
main {{ max-width:1460px; margin:26px auto; padding:0 22px 44px; }} .hero,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:24px; margin-bottom:20px; box-shadow:0 3px 12px rgba(20,40,65,.035); }}
.hero p {{ color:var(--muted); max-width:950px; margin-bottom:0; }} h2 {{ margin:0 0 14px; font-size:22px; }} h3 {{ margin:25px 0 12px; font-size:17px; }} .muted,.path {{ color:var(--muted); font-size:13px; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:13px; margin:18px 0; }} .metric {{ background:linear-gradient(140deg,#f8fbfd,#f0f6fa); border:1px solid #e0eaf0; border-radius:12px; padding:18px; min-height:130px; }} .metric-name {{ color:#496174; font-weight:650; font-size:13px; }} .metric-value {{ font-size:29px; font-weight:760; letter-spacing:-.02em; margin:9px 0 5px; }} .metric-desc {{ color:#738394; font-size:12px; }}
.data-table {{ width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }} .data-table-wrap {{ width:100%; overflow-x:auto; }} .data-table th {{ background:#203b55; color:white; font-weight:650; padding:12px 13px; white-space:nowrap; text-align:center; }} .data-table th:first-child {{ border-radius:8px 0 0 0; }} .data-table th:last-child {{ border-radius:0 8px 0 0; }} .data-table td {{ padding:11px 13px; text-align:center; border-bottom:1px solid #e8edf2; white-space:nowrap; }} .data-table tr:nth-child(even) td {{ background:#f7f9fc; }} .data-table tr:last-child td {{ font-weight:750; background:#eaf3f7; border-top:2px solid #c5dae4; }}
.report-image {{ display:block; max-width:100%; height:auto; margin:10px auto; border-radius:10px; }} details {{ border:1px solid var(--line); border-radius:10px; padding:13px 15px; margin:12px 0; }} summary {{ cursor:pointer; font-weight:650; }} .patch {{ border-top:1px solid var(--line); padding-top:20px; margin-top:20px; }} .patch-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:15px; }} .visual-card {{ min-width:0; background:#f6f8fb; border:1px solid var(--line); border-radius:10px; padding:10px; }} .visual-title {{ font-weight:700; font-size:13px; margin:0 0 8px; }} .patch-image {{ display:block; width:100%; height:auto; border-radius:6px; }}
.actions {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:18px; }} .action {{ color:#075a72; border:1px solid #c8dfe8; border-radius:10px; padding:12px 16px; text-decoration:none; font-weight:700; }} .action.primary {{ background:#0b647e; border-color:#0b647e; color:white; }} .notice {{ background:#fff8e8; border:1px solid #efddb0; padding:13px 16px; border-radius:10px; color:#6e5620; font-size:13px; }} footer {{ color:var(--muted); font-size:12px; padding:8px 3px; }}
@media(max-width:800px) {{ main {{ padding:0 10px 30px; margin-top:14px; }} .hero,.panel {{ padding:17px; }} .patch-grid {{ grid-template-columns:1fr; }} .nav {{ position:static; padding-left:12px; padding-right:12px; }} .metric-value {{ font-size:25px; }} }}
</style></head><body>
<header class="topbar"><div class="brand">Alaudah benchmark · 3D U-Net evaluation</div><h1>{html.escape(title)}</h1></header>
<nav class="nav">{links}</nav><main>{content}<footer>Отчёт сгенерирован локальным скриптом create_alaudah_report.py. Существующие файлы и visualizations не изменяются.</footer></main>
</body></html>'''


def count_distribution(args: argparse.Namespace) -> tuple[pd.DataFrame | None, str]:
    train = count_labels_h5(args.train_h5)
    val = count_labels_h5(args.val_h5)
    t1 = count_labels_npy(args.test1_dir / "ground_truth_volume.npy")
    if t1 is None:
        t1 = count_labels_h5(args.test1_h5)
    t2 = count_labels_npy(args.test2_dir / "ground_truth_volume.npy")
    if t2 is None:
        t2 = count_labels_h5(args.test2_h5)
    if any(x is None for x in (train, val, t1, t2)):
        return None, ""
    test = t1 + t2
    df = pd.DataFrame({
        "Class": np.arange(N_CLASSES),
        "TRAIN": percentages(train),
        "VAL": percentages(val),
        "TEST1": percentages(t1),
        "TEST2": percentages(t2),
        "TEST": percentages(test),
        "TRAIN_pixels": train, "VAL_pixels": val,
        "TEST1_pixels": t1, "TEST2_pixels": t2, "TEST_pixels": test,
    })
    return df, "generated from HDF5 labels and saved test ground-truth volumes"


def save_combined_artifacts(cm: np.ndarray, out_dir: Path, source1: str, source2: str) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, class_df = metrics_from_cm(cm)
    class_with_mean = add_mean_row(class_df, summary)
    np.savetxt(out_dir / "combined_confusion_matrix_counts.csv", cm, delimiter=",", fmt="%d")
    row_sums = cm.sum(axis=1, keepdims=True)
    percent_cm = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    np.savetxt(out_dir / "combined_confusion_matrix_percent.csv", percent_cm, delimiter=",", fmt="%.4f")
    class_df.to_csv(out_dir / "combined_class_metrics.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"test": "combined_test1_plus_test2", **summary,
                   "confusion_source_test1": source1, "confusion_source_test2": source2}]).to_csv(
        out_dir / "combined_summary.csv", index=False, float_format="%.8f")
    return summary


def make_split_page(title: str, slug: str, cm: np.ndarray, source: str, folder: Path,
                    summary: dict[str, float], class_df: pd.DataFrame, max_patches: int) -> str:
    detailed = add_mean_row(class_df, summary)
    warning = ""
    if "ПРИБЛИЖЁННО" in source or "APPROXIMATE" in source:
        warning = '<div class="notice">Внимание: матрица восстановлена из округлённых процентов. Метрики и объединённый результат могут быть приблизительными. Для точного объединения повторите тест с сохранением ground_truth_volume.npy и prediction_volume.npy.</div>'
    matrix_plot_uri = plot_confusion(cm, f"Матрица ошибок · {title}")
    row_sums = cm.sum(axis=1, keepdims=True)
    percent = np.divide(cm * 100.0, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    cm_table = pd.DataFrame(percent, columns=[f"Предсказан {i}" for i in range(N_CLASSES)])
    cm_table.insert(0, "Истинный класс", range(N_CLASSES))
    cm_table_html = cm_table.round(2).to_html(index=False, classes="data-table", border=0, escape=True)
    content = f'''<section class="hero"><p>Результат одного тестового объёма. Матрица ошибок: {html.escape(source)}.</p>
<p class="path">Папка результатов: <code>{html.escape(str(folder))}</code></p>{warning}</section>
<section class="panel"><h2>Ключевые метрики</h2><div class="metrics">{summary_cards(summary)}</div></section>
<section class="panel"><h2>Метрики по классам</h2><p class="muted">В последней строке среднее арифметическое class-wise метрик; частота и support суммируются. Вклад в FWIoU суммируется в общий FWIoU.</p><div class="data-table-wrap">{format_class_table(detailed)}</div></section>
<section class="panel"><h2>Матрица ошибок</h2>{image_from_path_data(matrix_plot_uri, 'Матрица ошибок')}
<details><summary>Открыть матрицу процентов таблицей</summary><div class="data-table-wrap">{cm_table_html}</div></details></section>
<section class="panel"><h2>Визуализации патчей</h2><p class="muted">Изображения читаются из существующей папки и не изменяются.</p>{visualizations_html(folder, max_patches)}</section>'''
    return page_shell(title, slug, content)


def image_from_path_data(uri: str | None, alt: str) -> str:
    if uri is None:
        return '<p class="muted">Матрица ошибок недоступна.</p>'
    return f'<img class="report-image" alt="{html.escape(alt)}" src="{uri}">'


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.combined_dir.mkdir(parents=True, exist_ok=True)

    # Load separate confusion matrices. Arrays are the preferred exact source.
    for label, path in (("test1", args.test1_dir), ("test2", args.test2_dir)):
        if not path.is_dir():
            raise SystemExit(f"Папка {label} не найдена: {path}\nУкажи реальный путь через --{label}_dir.")
    cm1, source1 = load_confusion(args.test1_dir)
    cm2, source2 = load_confusion(args.test2_dir)
    combined_cm = cm1 + cm2

    sum1, cls1 = metrics_from_cm(cm1)
    sum2, cls2 = metrics_from_cm(cm2)
    combined_summary = save_combined_artifacts(combined_cm, args.combined_dir, source1, source2)
    sum1["evaluated_voxels"] = float(cm1.sum())
    sum2["evaluated_voxels"] = float(cm2.sum())
    summaries = {"test1": sum1, "test2": sum2, "combined": combined_summary}
    combined_class_df = metrics_from_cm(combined_cm)[1]

    # Overview page: combined results immediately visible, with concise side-by-side comparison.
    overview_content = f'''<section class="hero"><h2>Главный результат: test1 + test2</h2>
<p>Итог посчитан честно: матрицы ошибок test1 и test2 сложены по абсолютным количествам вокселей, после чего метрики рассчитаны заново. Это не среднее арифметическое метрик двух тестов.</p>
<div class="metrics">{summary_cards(combined_summary)}</div>
<div class="actions"><a class="action primary" href="combined.html">Подробный объединённый результат →</a><a class="action" href="test1.html">Открыть Test 1</a><a class="action" href="test2.html">Открыть Test 2</a><a class="action" href="training.html">Обучение и распределение классов</a></div>
</section>
<section class="panel"><h2>Сравнение трёх результатов</h2>
<div class="data-table-wrap">{summary_comparison_table(summaries)}</div>
<p class="muted">Для сравнения с публикацией основными сводными показателями являются Pixel Accuracy, Mean Class Accuracy и Frequency Weighted IoU. Mean IoU по всем шести классам приведён дополнительно.</p>
</section>
<section class="panel"><h2>Источники расчёта</h2><p><strong>Test 1:</strong> {html.escape(source1)}.</p><p><strong>Test 2:</strong> {html.escape(source2)}.</p>
<p>Объединённая матрица и CSV-таблицы сохранены в <code>{html.escape(str(args.combined_dir))}</code>. Страницы test1 и test2 показывают свои метрики отдельно, а страница «Итоговый результат» содержит объединённую матрицу и class metrics.</p></section>'''
    (args.output_dir / "index.html").write_text(page_shell("Обзор результатов", "index", overview_content), encoding="utf-8")

    test1_html = make_split_page("Test 1", "test1", cm1, source1, args.test1_dir, sum1, cls1, args.max_visual_patches)
    test2_html = make_split_page("Test 2", "test2", cm2, source2, args.test2_dir, sum2, cls2, args.max_visual_patches)
    combined_source = f"объединение матриц test1 ({source1}) и test2 ({source2})"
    combined_html = make_split_page("Объединённый результат: Test 1 + Test 2", "combined", combined_cm,
                                    combined_source, args.combined_dir, combined_summary,
                                    combined_class_df, 0)
    (args.output_dir / "test1.html").write_text(test1_html, encoding="utf-8")
    (args.output_dir / "test2.html").write_text(test2_html, encoding="utf-8")
    (args.output_dir / "combined.html").write_text(combined_html, encoding="utf-8")

    # Training and class-distribution page: keep it away from the test detail pages.
    train_loss = read_loss(args.train_loss_csv)
    val_loss = read_loss(args.val_loss_csv)
    loss_uri = loss_plot(args.train_loss_csv, args.val_loss_csv)
    if loss_uri is None:
        loss_html = '<p class="muted">График loss не создан: укажи корректные пути через --train_loss_csv и --val_loss_csv.</p>'
    else:
        loss_html = image_from_path_data(loss_uri, "Train and validation loss")

    dist_df, dist_source = count_distribution(args)
    dist_html = render_class_distribution(dist_df)
    dist_uri = distribution_plot(dist_df)
    if dist_df is not None:
        distribution_csv = args.output_dir / "class_distribution.csv"
        dist_df.to_csv(distribution_csv, index=False, float_format="%.6f")
        print(f"Class distribution CSV saved: {distribution_csv.resolve()}")
    training_content = f'''<section class="panel"><h2>Train / validation loss</h2>{loss_html}
<p class="muted">Train: {html.escape(str(args.train_loss_csv))}<br>Validation: {html.escape(str(args.val_loss_csv))}</p></section>
<section class="panel"><h2>Распределение классов</h2><p class="muted">Источник: {html.escape(dist_source or 'данные доступны не полностью')}.</p>
{image_from_path_data(dist_uri, 'Распределение классов')}<div class="data-table-wrap">{dist_html}</div></section>'''
    (args.output_dir / "training.html").write_text(page_shell("Обучение и данные", "training", training_content), encoding="utf-8")

    print("\nHTML report created:")
    for name in ("index.html", "combined.html", "test1.html", "test2.html", "training.html"):
        print(f"  { (args.output_dir / name).resolve() }")
    print(f"Combined metrics CSVs saved in: {args.combined_dir.resolve()}")
    print("Open index.html in a browser to start from the overview page.")


if __name__ == "__main__":
    main()
