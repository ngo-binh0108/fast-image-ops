"""
Box (mean) filter in O(1) per pixel using running sums.

Windows are clipped at the image border and the mean is taken over the pixels
that fall inside the image, so there is no padding bias near the edges.
"""
import numpy as np


def _radii(radius):
    """Accept an int or a (ry, rx) pair and return two non-negative ints."""
    ry, rx = (radius, radius) if np.isscalar(radius) else radius
    ry, rx = int(ry), int(rx)
    if ry < 0 or rx < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    return ry, rx


def _window_bounds(n, radius):
    idx = np.arange(n)
    return np.maximum(idx - radius, 0), np.minimum(idx + radius + 1, n)


def window_sum(a, radius, axis):
    """Sum of a[i - radius : i + radius + 1] along `axis`, clipped to the array."""
    a = np.asarray(a)
    n = a.shape[axis]
    c = np.cumsum(a, axis=axis, dtype=np.float64)
    zero_shape = list(c.shape)
    zero_shape[axis] = 1
    c = np.concatenate([np.zeros(zero_shape), c], axis=axis)
    lo, hi = _window_bounds(n, radius)
    return np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)


def box_sum(img, radius):
    """Sum over a (2*ry+1) x (2*rx+1) window on the first two axes."""
    ry, rx = _radii(radius)
    return window_sum(window_sum(img, ry, 0), rx, 1)


def box_mean(img, radius):
    """
    Mean over a (2*ry+1) x (2*rx+1) window on the first two axes.

    Any trailing axes (channels, or the k x k blocks the guided filter uses) are
    filtered independently. Cost does not depend on the radius.
    """
    img = np.asarray(img)
    if img.ndim < 2:
        raise ValueError(f"expected at least a 2-D array, got shape {img.shape}")
    ry, rx = _radii(radius)
    h, w = img.shape[:2]
    lo, hi = _window_bounds(h, ry)
    count_y = hi - lo
    lo, hi = _window_bounds(w, rx)
    count_x = hi - lo
    count = np.outer(count_y, count_x).reshape((h, w) + (1,) * (img.ndim - 2))
    return box_sum(img, (ry, rx)) / count
