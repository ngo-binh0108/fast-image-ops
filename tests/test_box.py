import numpy as np
import pytest

from fast_image_ops import box_mean, box_sum


def brute_box_mean(img, ry, rx):
    h, w = img.shape[:2]
    out = np.empty(img.shape, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            win = img[max(y - ry, 0):y + ry + 1, max(x - rx, 0):x + rx + 1]
            out[y, x] = win.mean(axis=(0, 1))
    return out


@pytest.mark.parametrize("radius", [0, 1, 2, 5, (1, 3), (4, 0), 40])
def test_matches_brute_force(radius):
    rng = np.random.default_rng(0)
    img = rng.random((13, 17))
    ry, rx = (radius, radius) if np.isscalar(radius) else radius
    np.testing.assert_allclose(box_mean(img, radius), brute_box_mean(img, ry, rx), atol=1e-12)


def test_channels_are_filtered_independently():
    rng = np.random.default_rng(1)
    img = rng.random((9, 11, 3))
    out = box_mean(img, 2)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], box_mean(img[..., c], 2), atol=1e-12)


def test_integer_input_and_box_sum():
    img = np.arange(20, dtype=np.uint8).reshape(4, 5)
    assert box_sum(img, 10).max() == img.sum()
    np.testing.assert_allclose(box_mean(img, 10), np.full(img.shape, img.mean()))


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        box_mean(np.zeros(5), 1)
    with pytest.raises(ValueError):
        box_mean(np.zeros((5, 5)), -1)
