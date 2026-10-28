import numpy as np
import pytest
from scipy import ndimage

from fast_image_ops import dilate, erode, opening, closing, gradient, white_tophat, black_tophat


def size(radius):
    ry, rx = (radius, radius) if np.isscalar(radius) else radius
    return (2 * ry + 1, 2 * rx + 1)


RADII = [0, 1, 2, 3, (2, 5), (0, 4), 30]


@pytest.mark.parametrize("radius", RADII)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.uint8, np.int16])
def test_dilate_erode_match_scipy(radius, dtype):
    rng = np.random.default_rng(0)
    img = (rng.random((23, 31)) * 200).astype(dtype)
    np.testing.assert_array_equal(dilate(img, radius), ndimage.grey_dilation(img, size=size(radius), mode='nearest'))
    np.testing.assert_array_equal(erode(img, radius), ndimage.grey_erosion(img, size=size(radius), mode='nearest'))
    assert dilate(img, radius).dtype == dtype


@pytest.mark.parametrize("radius", [1, 3, (1, 4)])
def test_opening_closing_match_scipy(radius):
    rng = np.random.default_rng(1)
    img = rng.random((25, 19))
    np.testing.assert_array_equal(opening(img, radius), ndimage.grey_opening(img, size=size(radius), mode='nearest'))
    np.testing.assert_array_equal(closing(img, radius), ndimage.grey_closing(img, size=size(radius), mode='nearest'))


def test_gradient_and_tophats_do_not_wrap_for_uint8():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    f = img.astype(np.float64)
    np.testing.assert_array_equal(gradient(img, 2), dilate(f, 2) - erode(f, 2))
    np.testing.assert_array_equal(white_tophat(img, 2), f - opening(f, 2))
    np.testing.assert_array_equal(black_tophat(img, 2), closing(f, 2) - f)
    assert gradient(img, 2).min() >= 0


def test_bool_mask_matches_binary_dilation():
    rng = np.random.default_rng(3)
    mask = rng.random((30, 30)) > 0.97
    expected = ndimage.binary_dilation(mask, structure=np.ones((5, 5), bool))
    np.testing.assert_array_equal(dilate(mask, 2), expected)


def test_channels_last():
    rng = np.random.default_rng(4)
    img = rng.random((12, 14, 3))
    out = dilate(img, 2)
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], dilate(img[..., c], 2))
