import itertools

import numpy as np

from monotonic_decode import monotonic_decode


def _bruteforce(q):
    C, H = q.shape
    return max(
        itertools.combinations_with_replacement(range(C), H),
        key=lambda s: sum(np.log(q[c, i]) for i, c in enumerate(s)),
    )


def test_output_is_non_decreasing_with_depth():
    rng = np.random.default_rng(1)
    p = rng.random((6, 3, 40, 4))
    p /= p.sum(0, keepdims=True)
    labels = monotonic_decode(p)
    assert labels.shape == (3, 40, 4)
    assert (np.diff(labels, axis=1) >= 0).all()


def test_matches_bruteforce_optimum():
    rng = np.random.default_rng(2)
    for _ in range(5):
        p = rng.random((4, 1, 7, 1))
        p /= p.sum(0, keepdims=True)
        got = tuple(monotonic_decode(p)[0, :, 0])
        assert got == _bruteforce(p[:, 0, :, 0])


def test_keeps_monotonic_argmax_unchanged():
    C, H = 5, 50
    gt = np.repeat(np.arange(C), H // C)
    p = np.full((C, 1, H, 1), 0.05)
    p[gt, 0, np.arange(H), 0] = 0.8
    assert (monotonic_decode(p)[0, :, 0] == gt).all()
