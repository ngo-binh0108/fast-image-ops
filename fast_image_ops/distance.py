"""
Exact Euclidean distance transform in linear time
(Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions",
Theory of Computing 2012).

The squared 2-D transform separates into 1-D transforms along rows and then
columns. Each 1-D transform is the lower envelope of parabolas
(q - p)^2 + f(p). The envelope is built left to right with a stack, which is
inherently sequential along a line, so here the loop runs over positions and
every step is vectorized across all lines at once: O(n) Python iterations for
an n x n image instead of O(n^2).

Two parts need no Python loop at all. The first (row) pass of a binary mask
only has f in {0, inf}, so it is the distance to the nearest zero in the row,
found with two running `accumulate` scans. And reading the envelope back is a
single `searchsorted` over all lines, since every line's breakpoints are sorted.
"""
import numpy as np


def squared_distance_1d(f, axis=-1, spacing=1.0):
    """
    d(q) = min_p (spacing * (q - p))^2 + f(p) along `axis`, for every line of `f`.

    `spacing` is the distance between neighbouring samples. Entries of f may be
    +inf (no parabola there). Lines that are all +inf give +inf everywhere.
    """
    spacing = _positive_spacing(spacing)
    f = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    shape = f.shape
    n = shape[-1]
    f = f.reshape(-1, n)
    lines = f.shape[0]
    rows = np.arange(lines)

    # v[r, j]: position of the j-th envelope parabola; z[r, j]: where it starts
    v = np.zeros((lines, n), dtype=np.intp)
    z = np.full((lines, n + 1), np.inf)
    top = np.full(lines, -1, dtype=np.intp)

    # positions in physical units; breakpoints z are in the same units
    x = np.arange(n) * spacing

    def intersection(idx, q, p):
        xq, xp = x[q], x[p]
        return ((f[idx, q] + xq * xq) - (f[idx, p] + xp * xp)) / (2.0 * (xq - xp))

    for q in range(n):
        active = np.flatnonzero(np.isfinite(f[:, q]))
        if active.size == 0:
            continue
        # pop parabolas that the new one hides completely
        idx = active[top[active] >= 0]
        while idx.size:
            s = intersection(idx, q, v[idx, top[idx]])
            idx = idx[s <= z[idx, top[idx]]]
            top[idx] -= 1
            idx = idx[top[idx] >= 0]
        k = top[active]
        s = np.full(active.size, -np.inf)
        nonempty = k >= 0
        s[nonempty] = intersection(active[nonempty], q, v[active[nonempty], k[nonempty]])
        top[active] = k + 1
        v[active, k + 1] = q
        z[active, k + 1] = s
        z[active, k + 2] = np.inf

    # The parabola covering q is the j-th one, where j counts the breakpoints
    # z[r, 1:] below x[q]. Clipping z to [-spacing, n * spacing] keeps each
    # line's breakpoints sorted, and offsetting line r by r * (n + 2) * spacing
    # makes the whole array sorted, so one searchsorted finds j for every
    # (line, q) at once.
    offsets = rows[:, None] * ((n + 2) * spacing)
    breaks = (np.clip(z[:, 1:], -spacing, n * spacing) + offsets).ravel()
    j = np.searchsorted(breaks, (x + offsets).ravel(), side='left').reshape(lines, n) - rows[:, None] * n
    p = v[rows[:, None], j]
    d = (x - x[p]) ** 2 + f[rows[:, None], p]
    return np.moveaxis(d.reshape(shape), -1, axis)


def _positive_spacing(spacing):
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"spacing must be a positive number, got {spacing}")
    return spacing


def _nearest_zero_1d(zero, spacing=1.0):
    """Squared distance along the last axis to the nearest True of `zero` (inf if none)."""
    n = zero.shape[-1]
    idx = np.arange(n)
    left = np.maximum.accumulate(np.where(zero, idx, -2 * n), axis=-1)
    right = np.minimum.accumulate(np.where(zero, idx, 3 * n)[..., ::-1], axis=-1)[..., ::-1]
    d = np.minimum(idx - left, right - idx).astype(np.float64)
    d[d > n] = np.inf
    d *= spacing
    return d * d


def distance_transform(mask, squared=False, sampling=None):
    """
    Euclidean distance from every pixel to the nearest zero (False) pixel of
    the 2-D `mask`, matching scipy.ndimage.distance_transform_edt. Zero pixels
    get 0. If the mask has no zero pixels the result is +inf everywhere.

    sampling: pixel size, a number or (row spacing, column spacing), for
              distances in physical units (e.g. metres for a raster with
              10 m x 20 m pixels). Defaults to 1 for both axes.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if sampling is None:
        sampling = 1.0
    sy, sx = (sampling, sampling) if np.isscalar(sampling) else sampling
    sy, sx = _positive_spacing(sy), _positive_spacing(sx)
    d2 = squared_distance_1d(_nearest_zero_1d(mask == 0, sx), axis=0, spacing=sy)
    return d2 if squared else np.sqrt(d2)
