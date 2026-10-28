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
"""
import numpy as np


def squared_distance_1d(f, axis=-1):
    """
    d(q) = min_p (q - p)^2 + f(p) along `axis`, for every line of `f`.

    Entries of f may be +inf (no parabola there). Lines that are all +inf
    give +inf everywhere.
    """
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

    def intersection(idx, q, p):
        return ((f[idx, q] + q * q) - (f[idx, p] + p * p)) / (2.0 * (q - p))

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

    d = np.empty_like(f)
    j = np.zeros(lines, dtype=np.intp)
    for q in range(n):
        while True:
            advance = z[rows, j + 1] < q
            if not advance.any():
                break
            j[advance] += 1
        p = v[rows, j]
        d[:, q] = (q - p) ** 2 + f[rows, p]
    return np.moveaxis(d.reshape(shape), -1, axis)


def distance_transform(mask, squared=False):
    """
    Euclidean distance from every pixel to the nearest zero (False) pixel of
    the 2-D `mask`, matching scipy.ndimage.distance_transform_edt. Zero pixels
    get 0. If the mask has no zero pixels the result is +inf everywhere.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    f = np.where(mask != 0, np.inf, 0.0)
    d2 = squared_distance_1d(squared_distance_1d(f, axis=1), axis=0)
    return d2 if squared else np.sqrt(d2)
