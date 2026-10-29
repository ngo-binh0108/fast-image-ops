import numpy as np
import pytest
from scipy import ndimage

from fast_image_ops import distance_transform, squared_distance_1d


@pytest.mark.parametrize("density", [0.001, 0.02, 0.3, 0.9])
@pytest.mark.parametrize("shape", [(1, 40), (37, 1), (41, 53), (64, 64)])
def test_matches_scipy(density, shape):
    rng = np.random.default_rng(0)
    mask = rng.random(shape) > density
    mask.flat[rng.integers(mask.size)] = False  # at least one zero
    d2 = distance_transform(mask, squared=True)
    expected = ndimage.distance_transform_edt(mask)
    # squared distances are integers, so the comparison can be exact
    np.testing.assert_array_equal(d2, np.round(expected ** 2))
    np.testing.assert_allclose(distance_transform(mask), expected, rtol=0, atol=1e-12)


def test_brute_force_small():
    rng = np.random.default_rng(1)
    mask = rng.random((9, 12)) > 0.15
    zeros = np.argwhere(~mask)
    ys, xs = np.mgrid[0:9, 0:12]
    brute = ((ys[..., None] - zeros[:, 0]) ** 2 + (xs[..., None] - zeros[:, 1]) ** 2).min(axis=-1)
    np.testing.assert_array_equal(distance_transform(mask, squared=True), brute)


def test_single_point_gives_radial_distance():
    mask = np.ones((21, 21), bool)
    mask[10, 10] = False
    ys, xs = np.mgrid[0:21, 0:21]
    np.testing.assert_allclose(distance_transform(mask), np.hypot(ys - 10, xs - 10))


def test_no_zeros_gives_inf():
    assert np.isinf(distance_transform(np.ones((5, 6)))).all()
    assert (distance_transform(np.zeros((5, 6))) == 0).all()


def test_1d_lower_envelope_against_brute_force():
    rng = np.random.default_rng(2)
    f = rng.random((6, 30)) * 50
    f[rng.random(f.shape) < 0.5] = np.inf
    f[0] = np.inf  # a line with no parabolas at all
    q = np.arange(30)
    brute = ((q[:, None] - q[None, :]) ** 2 + f[:, None, :]).min(axis=-1)
    np.testing.assert_allclose(squared_distance_1d(f), brute)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        distance_transform(np.ones((2, 2, 2)))


@pytest.mark.parametrize("sampling", [2.0, (1.0, 2.5), (10.0, 20.0), (0.3, 1.7)])
def test_sampling_matches_scipy(sampling):
    rng = np.random.default_rng(3)
    mask = rng.random((40, 57)) > 0.02
    expected = ndimage.distance_transform_edt(mask, sampling=sampling)
    np.testing.assert_allclose(distance_transform(mask, sampling=sampling), expected, rtol=1e-12, atol=1e-9)


def test_sampling_scales_uniformly():
    rng = np.random.default_rng(4)
    mask = rng.random((30, 30)) > 0.05
    np.testing.assert_allclose(distance_transform(mask, sampling=10.0), 10.0 * distance_transform(mask))


def test_1d_spacing_against_brute_force():
    rng = np.random.default_rng(5)
    f = rng.random((4, 25)) * 50
    f[rng.random(f.shape) < 0.5] = np.inf
    x = np.arange(25) * 1.7
    brute = ((x[:, None] - x[None, :]) ** 2 + f[:, None, :]).min(axis=-1)
    np.testing.assert_allclose(squared_distance_1d(f, spacing=1.7), brute)


@pytest.mark.parametrize("sampling", [0, -1.0, (1.0, 0.0), float('nan')])
def test_rejects_bad_sampling(sampling):
    with pytest.raises(ValueError):
        distance_transform(np.ones((3, 3)), sampling=sampling)
