"""Монотонное декодирование: на каждой трассе метки не убывают с глубиной.

probs: [C, D, H, W] — усреднённые softmax после склейки (ось H = глубина).
Возвращает метки [D, H, W], максимизирующие сумму log p при условии
label[h] <= label[h+1]. Динамическое программирование O(H * C) на трассу,
векторизовано по всем трассам сразу.
"""
import numpy as np


def monotonic_decode(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    C, D, H, W = probs.shape
    logp = np.log(np.clip(probs, eps, 1.0)).transpose(2, 0, 1, 3).reshape(H, C, D * W)
    n = D * W
    score = logp[0].copy()                      # [C, N]
    back = np.zeros((H, C, n), dtype=np.int8)   # откуда пришли
    for h in range(1, H):
        # лучший предыдущий класс <= c: префиксный максимум по классам
        best_val = np.maximum.accumulate(score, axis=0)
        # индекс префиксного максимума
        idx = np.zeros_like(score, dtype=np.int64)
        cur = np.zeros(n, dtype=np.int64)
        for c in range(1, C):
            better = score[c] > best_val[c - 1]
            cur = np.where(better, c, cur)
            idx[c] = cur
        back[h] = idx
        score = best_val + logp[h]
    labels = np.empty((H, n), dtype=np.int64)
    labels[-1] = score.argmax(axis=0)
    cols = np.arange(n)
    for h in range(H - 1, 0, -1):
        labels[h - 1] = back[h, labels[h], cols]
    return labels.reshape(H, D, W).transpose(1, 0, 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    C, D, H, W = 5, 2, 60, 3
    gt = np.repeat(np.repeat(np.repeat(np.arange(C), H // C)[None, :, None], D, 0), W, 2)
    p = rng.random((C, D, H, W)) * 0.6
    for c in range(C):
        p[c][gt == c] += 0.5
    p /= p.sum(0, keepdims=True)
    am = p.argmax(0)
    mo = monotonic_decode(p)
    assert (np.diff(mo, axis=1) >= 0).all()
    # проверка оптимальности перебором на одной короткой трассе
    import itertools
    q = p[:, 0, :8, 0]
    best = max((s for s in itertools.combinations_with_replacement(range(C), 8)),
               key=lambda s: sum(np.log(q[c, i]) for i, c in enumerate(s)))
    got = monotonic_decode(p[:, :1, :8, :1])[0, :, 0]
    assert tuple(got) == best, (got, best)
    print("argmax acc", (am == gt).mean(), "monotonic acc", (mo == gt).mean())
