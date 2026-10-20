import numpy as np
import pytest

pytest.importorskip("torch")
from pytorch3dunet.augment.transforms import RandomFlip  # noqa: E402


def test_random_flip_never_flips_depth():
    # (crossline, depth, inline); value encodes depth index
    m = np.broadcast_to(np.arange(12, dtype=np.float32)[None, :, None], (4, 12, 6)).copy()
    for seed in range(50):
        out = RandomFlip(np.random.RandomState(seed))(m)
        assert (out[:, :, :] == m).all(), "depth order changed"


def test_random_flip_still_flips_lateral_axes():
    m = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    flipped = [not np.array_equal(RandomFlip(np.random.RandomState(s))(m), m) for s in range(20)]
    assert any(flipped)
