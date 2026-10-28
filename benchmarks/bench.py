"""
Time the filters against SciPy and against a naive O(r^2)-per-pixel window
scan, for growing radii.
$ python benchmarks/bench.py [size]
"""
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

import fast_image_ops as fio


def best_of(fn, repeat=3):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def naive(img, r, reduce):
    padded = np.pad(img, r, mode='edge')
    return reduce(sliding_window_view(padded, (2 * r + 1, 2 * r + 1)), axis=(-2, -1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rng = np.random.default_rng(0)
    img = rng.random((n, n))
    color = rng.random((n, n, 3))
    mask = rng.random((n, n)) > 0.01
    radii = [2, 8, 32]

    print(f"image {n}x{n} float64, best of 3, milliseconds\n")
    print(f"| filter | implementation | " + " | ".join(f"r={r}" for r in radii) + " |")
    print("|---|---|" + "---|" * len(radii))

    def row(name, impl, fn, max_naive_r=None):
        cells = []
        for r in radii:
            if max_naive_r is not None and r > max_naive_r:
                cells.append("-")
            else:
                cells.append(f"{best_of(lambda: fn(r)):.1f}")
        print(f"| {name} | {impl} | " + " | ".join(cells) + " |")

    row("box mean", "fast_image_ops", lambda r: fio.box_mean(img, r))
    row("box mean", "scipy uniform_filter", lambda r: ndimage.uniform_filter(img, 2 * r + 1, mode='nearest'))
    row("box mean", "naive window scan", lambda r: naive(img, r, np.mean), max_naive_r=8)
    row("dilation", "fast_image_ops", lambda r: fio.dilate(img, r))
    row("dilation", "scipy grey_dilation", lambda r: ndimage.grey_dilation(img, size=(2 * r + 1,) * 2, mode='nearest'))
    row("dilation", "naive window scan", lambda r: naive(img, r, np.max), max_naive_r=8)
    row("guided, gray guide", "guided_filter", lambda r: fio.guided_filter(img, img, r, 1e-2))
    row("guided, gray guide", "fast_guided_filter s=4", lambda r: fio.fast_guided_filter(img, img, r, 1e-2, scale=4))
    row("guided, color guide", "guided_filter", lambda r: fio.guided_filter(color, img, r, 1e-2))
    row("guided, color guide", "fast_guided_filter s=4", lambda r: fio.fast_guided_filter(color, img, r, 1e-2, scale=4))

    ours = best_of(lambda: fio.distance_transform(mask))
    ref = best_of(lambda: ndimage.distance_transform_edt(mask))
    print(f"\nexact EDT (1% zeros): fast_image_ops {ours:.1f} ms, scipy distance_transform_edt {ref:.1f} ms")


if __name__ == '__main__':
    main()
