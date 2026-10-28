"""
Guided filter (He, Sun, Tang, "Guided Image Filtering", TPAMI 2013) and the
subsampled fast guided filter (He & Sun, arXiv:1505.00996, 2015).

In every window the output is modelled as a linear function of the guide,
q = a . I + b, fitted to the input p by ridge regression. Each step is a box
mean, so the cost per pixel is independent of the radius. The guide can be
grayscale (H, W) or color (H, W, K); the input can be (H, W) or (H, W, C).
"""
import numpy as np

from .box import box_mean, _radii


def _as_channels(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x[..., None], True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"expected an (H, W) or (H, W, C) array, got shape {x.shape}")


def _solve_sym3(s00, s01, s02, s11, s12, s22, rhs):
    """
    Solve S a = rhs per pixel for symmetric 3 x 3 S given by its six (H, W)
    components and rhs of shape (H, W, 3, C), using the cofactor inverse.
    Working on whole-image component arrays avoids the per-pixel overhead of
    np.linalg.solve and of batched 3 x 3 matmuls.
    """
    i00 = s11 * s22 - s12 * s12
    i01 = s02 * s12 - s01 * s22
    i02 = s01 * s12 - s02 * s11
    i11 = s00 * s22 - s02 * s02
    i12 = s01 * s02 - s00 * s12
    i22 = s00 * s11 - s01 * s01
    inv_det = 1.0 / (s00 * i00 + s01 * i01 + s02 * i02)
    i00, i01, i02, i11, i12, i22 = (x[..., None] * inv_det[..., None] for x in (i00, i01, i02, i11, i12, i22))
    r0, r1, r2 = rhs[..., 0, :], rhs[..., 1, :], rhs[..., 2, :]
    a = np.empty_like(rhs)
    a[..., 0, :] = i00 * r0 + i01 * r1 + i02 * r2
    a[..., 1, :] = i01 * r0 + i11 * r1 + i12 * r2
    a[..., 2, :] = i02 * r0 + i12 * r1 + i22 * r2
    return a


def _linear_coefficients(I, p, radius, eps):
    """Per-window coefficients a (H, W, K, C) and b (H, W, C) of q = a . I + b."""
    h, w, k = I.shape
    c = p.shape[-1]
    iu = np.triu_indices(k)
    # every per-pixel term goes through a single box filter call:
    # I (K), p (C), the upper triangle of I I^T, and I p^T (K * C)
    terms = np.concatenate([I, p, I[..., iu[0]] * I[..., iu[1]],
                            (I[..., :, None] * p[..., None, :]).reshape(h, w, k * c)], axis=-1)
    means = box_mean(terms, radius)
    mean_I, mean_p = means[..., :k], means[..., k:k + c]
    mean_II = means[..., k + c:k + c + len(iu[0])]
    mean_Ip = means[..., k + c + len(iu[0]):].reshape(h, w, k, c)
    cov_Ip = mean_Ip - mean_I[..., :, None] * mean_p[..., None, :]
    # covariance of the guide, upper triangle in triu_indices order, eps on the diagonal
    var_I = [mean_II[..., t] - mean_I[..., i] * mean_I[..., j] + (eps if i == j else 0.0)
             for t, (i, j) in enumerate(zip(*iu))]
    if k == 1:
        a = cov_Ip / var_I[0][..., None, None]
    elif k == 3:
        a = _solve_sym3(*var_I, cov_Ip)
    else:
        sigma = np.empty((h, w, k, k))
        for t, (i, j) in enumerate(zip(*iu)):
            sigma[..., i, j] = sigma[..., j, i] = var_I[t]
        a = np.linalg.solve(sigma, cov_Ip)
    b = mean_p - np.einsum('...kc,...k->...c', a, mean_I)
    return a, b


def _apply(a, b, I, radius):
    """q = mean(a) . I + mean(b), with both means in one box filter call."""
    h, w, k, c = a.shape
    means = box_mean(np.concatenate([a.reshape(h, w, k * c), b], axis=-1), radius)
    return means[..., :k * c].reshape(h, w, k, c), means[..., k * c:]


def guided_filter(guide, src, radius, eps):
    """
    Edge-preserving smoothing of `src` steered by `guide`.

    radius: window radius (int or (ry, rx)); the window is (2r+1) x (2r+1).
    eps:    regularization; larger values smooth more. For images in [0, 1]
            values around 1e-3 .. 1e-1 are typical.
    """
    I, _ = _as_channels(guide)
    p, squeeze = _as_channels(src)
    if I.shape[:2] != p.shape[:2]:
        raise ValueError(f"guide {I.shape[:2]} and src {p.shape[:2]} sizes differ")
    mean_a, mean_b = _apply(*_linear_coefficients(I, p, radius, eps), I, radius)
    q = np.einsum('...kc,...k->...c', mean_a, I) + mean_b
    return q[..., 0] if squeeze else q


def resize_bilinear(x, shape):
    """Bilinear resize of the first two axes to `shape`, pixel centers aligned."""
    x = np.asarray(x, dtype=np.float64)
    h, w = x.shape[:2]
    H, W = shape

    def axis_weights(n_in, n_out):
        pos = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        pos = np.clip(pos, 0, n_in - 1)
        i0 = np.floor(pos).astype(np.intp)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, pos - i0

    y0, y1, wy = axis_weights(h, H)
    x0, x1, wx = axis_weights(w, W)
    extra = (1,) * (x.ndim - 2)
    wy = wy.reshape((H, 1) + extra)
    wx = wx.reshape((1, W) + extra)
    top = x[y0][:, x0] * (1 - wx) + x[y0][:, x1] * wx
    bottom = x[y1][:, x0] * (1 - wx) + x[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def fast_guided_filter(guide, src, radius, eps, scale=4):
    """
    Guided filter with the coefficients computed on a `scale`-times smaller
    image and upsampled. About scale**2 times less work for the box filters;
    the output stays sharp because the final a . I + b uses the full-res guide.
    """
    scale = int(scale)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return guided_filter(guide, src, radius, eps)
    I, _ = _as_channels(guide)
    p, squeeze = _as_channels(src)
    if I.shape[:2] != p.shape[:2]:
        raise ValueError(f"guide {I.shape[:2]} and src {p.shape[:2]} sizes differ")
    ry, rx = _radii(radius)
    small_radius = (max(ry // scale, 1) if ry else 0, max(rx // scale, 1) if rx else 0)
    small_shape = (-(-I.shape[0] // scale), -(-I.shape[1] // scale))
    I_small = resize_bilinear(I, small_shape)
    p_small = resize_bilinear(p, small_shape)
    mean_a, mean_b = _apply(*_linear_coefficients(I_small, p_small, small_radius, eps), I_small, small_radius)
    mean_a = resize_bilinear(mean_a, I.shape[:2])
    mean_b = resize_bilinear(mean_b, I.shape[:2])
    q = np.einsum('...kc,...k->...c', mean_a, I) + mean_b
    return q[..., 0] if squeeze else q
