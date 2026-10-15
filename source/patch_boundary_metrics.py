"""Ошибки на краю отдельного патча и в его центре.

Считается по предсказанию одного патча до усреднения перекрытий.
Это не метрика reconstructed volume и не замена официального mIoU.

Воксель, который лежит на краю одного патча и в центре другого,
попадает в обе группы. Знаменатель — число таких попаданий, не число
уникальных вокселей куба.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class PatchBoundaryStats:
    def __init__(self, n_classes: int = 10, border: int = 4, focus: tuple[int, ...] = (3, 4, 5, 6)):
        if border < 1:
            raise ValueError("border must be >= 1")
        self.n_classes = int(n_classes)
        self.border = int(border)
        self.focus = tuple(int(c) for c in focus)
        self.n_patches = 0
        self.confusion = {
            "edge": np.zeros((self.n_classes, self.n_classes), dtype=np.int64),
            "interior": np.zeros((self.n_classes, self.n_classes), dtype=np.int64),
        }

    def add_patch(self, prediction, ground_truth) -> None:
        pred = np.asarray(prediction)
        gt = np.asarray(ground_truth)
        if pred.shape != gt.shape or pred.ndim != 3:
            raise ValueError(f"Ожидались маски [D,H,W] одной формы, пришло pred={pred.shape}, gt={gt.shape}")
        depth, height, width = pred.shape
        if min(depth, height, width) <= 2 * self.border:
            raise ValueError(f"Патч {pred.shape} меньше, чем две полосы шириной {self.border}")

        edge = np.zeros(pred.shape, dtype=bool)
        edge[: self.border] = True
        edge[-self.border :] = True
        edge[:, : self.border] = True
        edge[:, -self.border :] = True
        edge[:, :, : self.border] = True
        edge[:, :, -self.border :] = True

        self._accumulate("edge", pred, gt, edge)
        self._accumulate("interior", pred, gt, ~edge)
        self.n_patches += 1

    def _accumulate(self, region: str, pred, gt, mask) -> None:
        true = gt[mask].astype(np.int64, copy=False).reshape(-1)
        guessed = pred[mask].astype(np.int64, copy=False).reshape(-1)
        valid = (true >= 0) & (true < self.n_classes) & (guessed >= 0) & (guessed < self.n_classes)
        true = true[valid]
        guessed = guessed[valid]
        index = self.n_classes * true + guessed
        counts = np.bincount(index, minlength=self.n_classes * self.n_classes)
        self.confusion[region] += counts.reshape(self.n_classes, self.n_classes)

    def rows(self) -> list[dict]:
        """IoU здесь считается только по вокселям региона, с повторами на перекрытиях."""
        out = []
        for region, matrix in self.confusion.items():
            tp = np.diag(matrix)
            support = matrix.sum(axis=1)
            predicted = matrix.sum(axis=0)
            union = tp + (predicted - tp) + (support - tp)
            total = int(matrix.sum())
            errors = int(total - tp.sum())
            out.append(self._row(region, "all", total, errors, _safe_div(errors, total), _safe_div(tp.sum(), union.sum())))
            for class_id in self.focus:
                class_total = int(support[class_id])
                class_errors = int(class_total - tp[class_id])
                out.append(
                    self._row(
                        region,
                        str(class_id),
                        class_total,
                        class_errors,
                        _safe_div(class_errors, class_total),
                        _safe_div(tp[class_id], union[class_id]),
                    )
                )
        return out

    def _row(self, region, class_name, support, errors, error_rate, iou) -> dict:
        return {
            "region": region,
            "class": class_name,
            "support": support,
            "errors": errors,
            "error_rate": error_rate,
            "iou": iou,
            "border_voxels": self.border,
            "n_patches": self.n_patches,
            "count_meaning": "patch_occurrences_not_unique_voxels",
        }

    def save_csv(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def summary_line(self) -> str:
        by_key = {(row["region"], row["class"]): row for row in self.rows()}
        edge = by_key[("edge", "all")]["error_rate"]
        interior = by_key[("interior", "all")]["error_rate"]
        return (
            f"patch edge error {edge:.4f}, interior {interior:.4f}, "
            f"edge-interior {edge - interior:+.4f} "
            f"(border={self.border}, patches={self.n_patches})"
        )


def _safe_div(numerator, denominator) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _self_check() -> None:
    stats = PatchBoundaryStats(n_classes=10, border=4)
    pred = np.zeros((16, 32, 32), dtype=np.int64)
    gt = np.zeros_like(pred)
    edge = np.zeros(pred.shape, dtype=bool)
    edge[:4] = edge[-4:] = True
    edge[:, :4] = edge[:, -4:] = True
    edge[:, :, :4] = edge[:, :, -4:] = True
    gt[:] = 6
    pred[:] = 6
    pred[edge] = 5
    stats.add_patch(pred, gt)
    by_key = {(row["region"], row["class"]): row for row in stats.rows()}
    if by_key[("edge", "all")]["error_rate"] != 1.0:
        raise AssertionError(by_key[("edge", "all")])
    if by_key[("interior", "all")]["error_rate"] != 0.0:
        raise AssertionError(by_key[("interior", "all")])
    if by_key[("edge", "6")]["errors"] != by_key[("edge", "all")]["errors"]:
        raise AssertionError("class 6 should own every edge error in the fixture")


if __name__ == "__main__":
    _self_check()
    print("self-check ok")
