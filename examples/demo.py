"""
Render assets/demo.png: every filter applied to a synthetic noisy scene.
$ python examples/demo.py
Needs matplotlib (pip install matplotlib).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import fast_image_ops as fio


def scene(n=256, seed=0):
    """Smooth background, sharp-edged shapes and a few small bright specks, plus noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.25 + 0.15 * np.sin(3 * x + 2 * y)
    img[(x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.04] = 0.85
    img[(x > 0.55) & (x < 0.85) & (y > 0.2) & (y < 0.75)] = 0.6
    img[(np.abs(x - 0.35) < 0.15) & (np.abs(y - 0.8) < 0.04)] = 0.05
    for py, px in rng.integers(10, n - 10, (12, 2)):
        img[py - 1:py + 2, px - 1:px + 2] = 1.0
    clean = img.copy()
    return clean, np.clip(img + rng.normal(0, 0.06, img.shape), 0, 1)


def main():
    clean, noisy = scene()
    panels = [
        ("noisy input", noisy, 'gray'),
        ("box mean r=4", fio.box_mean(noisy, 4), 'gray'),
        ("guided filter r=4, eps=0.01", fio.guided_filter(noisy, noisy, 4, 1e-2), 'gray'),
        ("fast guided filter r=8, s=4", fio.fast_guided_filter(noisy, noisy, 8, 1e-2, scale=4), 'gray'),
        ("white top-hat r=3 (small bright specks)", fio.white_tophat(clean, 3), 'magma'),
        ("morphological gradient r=1 (edges)", fio.gradient(clean, 1), 'magma'),
        ("mask: guided output > 0.5", fio.guided_filter(noisy, noisy, 4, 1e-2) > 0.5, 'gray'),
        ("distance to mask (exact EDT)", fio.distance_transform(fio.guided_filter(noisy, noisy, 4, 1e-2) <= 0.5), 'viridis'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
    for ax, (title, im, cmap) in zip(axes.flat, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    os.makedirs('assets', exist_ok=True)
    fig.savefig('assets/demo.png', dpi=90)
    print("saved assets/demo.png")


if __name__ == '__main__':
    main()
