import numpy as np
import pytest

from fast_image_ops import guided_filter, fast_guided_filter, box_mean, resize_bilinear


def brute_guided_filter(I, p, r, eps):
    """
    Direct definition: in every window solve the ridge regression
    min sum (a . I + b - p)^2 + N * eps * |a|^2, then average the per-window
    predictions over all windows that cover a pixel.
    """
    I = I[..., None] if I.ndim == 2 else I
    p2 = p[..., None] if p.ndim == 2 else p
    h, w, k = I.shape
    c = p2.shape[-1]
    a = np.empty((h, w, k, c))
    b = np.empty((h, w, c))
    for y in range(h):
        for x in range(w):
            sl = (slice(max(y - r, 0), y + r + 1), slice(max(x - r, 0), x + r + 1))
            X = I[sl].reshape(-1, k)
            n = X.shape[0]
            X1 = np.hstack([X, np.ones((n, 1))])
            reg = n * eps * np.diag([1.0] * k + [0.0])
            theta = np.linalg.solve(X1.T @ X1 + reg, X1.T @ p2[sl].reshape(-1, c))
            a[y, x], b[y, x] = theta[:k], theta[k]
    q = np.einsum('...kc,...k->...c', box_mean(a, r), I) + box_mean(b, r)
    return q[..., 0] if p.ndim == 2 else q


@pytest.mark.parametrize("r,eps", [(1, 1e-2), (2, 1e-3), (4, 1e-1)])
def test_gray_guide_matches_definition(r, eps):
    rng = np.random.default_rng(0)
    I = rng.random((12, 15))
    p = rng.random((12, 15))
    np.testing.assert_allclose(guided_filter(I, p, r, eps), brute_guided_filter(I, p, r, eps), atol=1e-10)


def test_color_guide_multichannel_src_matches_definition():
    rng = np.random.default_rng(1)
    I = rng.random((10, 11, 3))
    p = rng.random((10, 11, 2))
    np.testing.assert_allclose(guided_filter(I, p, 2, 1e-2), brute_guided_filter(I, p, 2, 1e-2), atol=1e-9)


def test_two_channel_guide_uses_generic_solver():
    rng = np.random.default_rng(7)
    I = rng.random((9, 10, 2))
    p = rng.random((9, 10))
    np.testing.assert_allclose(guided_filter(I, p, 1, 1e-2), brute_guided_filter(I, p, 1, 1e-2), atol=1e-9)


def test_self_guided_with_tiny_eps_is_identity():
    rng = np.random.default_rng(2)
    p = rng.random((20, 20))
    np.testing.assert_allclose(guided_filter(p, p, 3, 1e-12), p, atol=1e-6)


def test_huge_eps_is_double_box_mean():
    rng = np.random.default_rng(3)
    I, p = rng.random((16, 16)), rng.random((16, 16))
    np.testing.assert_allclose(guided_filter(I, p, 2, 1e12), box_mean(box_mean(p, 2), 2), atol=1e-9)


def test_preserves_step_edge():
    # a noisy step: the filter should flatten the noise but keep the jump
    rng = np.random.default_rng(4)
    clean = np.zeros((64, 64))
    clean[:, 32:] = 1.0
    noisy = clean + rng.normal(0, 0.05, clean.shape)
    out = guided_filter(noisy, noisy, 4, 1e-2)
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean() / 2
    assert out[:, 34:].mean() - out[:, :30].mean() > 0.95
    # a plain box blur of the same size smears the edge much more
    blur = box_mean(noisy, 4)
    assert np.abs(out[:, 28:36] - clean[:, 28:36]).mean() < np.abs(blur[:, 28:36] - clean[:, 28:36]).mean() / 3


def test_fast_scale_1_equals_full():
    rng = np.random.default_rng(5)
    I, p = rng.random((20, 24, 3)), rng.random((20, 24))
    np.testing.assert_array_equal(fast_guided_filter(I, p, 3, 1e-2, scale=1), guided_filter(I, p, 3, 1e-2))


def test_fast_is_close_to_full_on_smooth_images():
    y, x = np.mgrid[0:128, 0:128] / 128.0
    I = np.sin(6 * x) * np.cos(4 * y) * 0.5 + 0.5
    p = I + 0.1 * np.sin(20 * x)
    full = guided_filter(I, p, 8, 1e-3)
    fast = fast_guided_filter(I, p, 8, 1e-3, scale=4)
    assert np.abs(fast - full).mean() < 0.01


def test_resize_bilinear_identity_and_constant():
    rng = np.random.default_rng(6)
    x = rng.random((7, 9, 2))
    np.testing.assert_allclose(resize_bilinear(x, (7, 9)), x)
    np.testing.assert_allclose(resize_bilinear(np.full((5, 5), 3.0), (13, 8)), 3.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        guided_filter(np.zeros((4, 4)), np.zeros((4, 5)), 1, 1e-2)
