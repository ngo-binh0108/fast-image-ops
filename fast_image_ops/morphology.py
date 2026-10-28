"""
Grayscale morphology with rectangular structuring elements using the
van Herk / Gil-Werman algorithm: about three comparisons per pixel per axis,
whatever the window size.

The 1-D trick: split the (padded) signal into blocks of the window length k.
Any window of length k spans at most two neighbouring blocks, so its maximum is
max(suffix max of the first block from the window start, prefix max of the next
block up to the window end). Both running extrema are one `accumulate` call.

Pixels outside the image are ignored (equivalent to scipy.ndimage's
mode='nearest' for these filters).
"""
import numpy as np

from .box import _radii


def _identity(dtype, op):
    """The value that never wins for `op` (e.g. -inf for maximum)."""
    if dtype == np.bool_:
        return op is np.minimum
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min if op is np.maximum else info.max
    if np.issubdtype(dtype, np.floating):
        return -np.inf if op is np.maximum else np.inf
    raise TypeError(f"unsupported dtype {dtype}")


def sliding_extremum(a, radius, axis, op):
    """op over a[i - radius : i + radius + 1] along `axis` (op is np.maximum or np.minimum)."""
    a = np.asarray(a)
    if radius == 0:
        return a.copy()
    k = 2 * radius + 1
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    total = -(-(n + 2 * radius) // k) * k
    padded = np.full(a.shape[:-1] + (total,), _identity(a.dtype, op), dtype=a.dtype)
    padded[..., radius:radius + n] = a
    blocks = padded.reshape(a.shape[:-1] + (total // k, k))
    prefix = op.accumulate(blocks, axis=-1).reshape(padded.shape)
    suffix = op.accumulate(blocks[..., ::-1], axis=-1)[..., ::-1].reshape(padded.shape)
    out = op(suffix[..., :n], prefix[..., 2 * radius:2 * radius + n])
    return np.moveaxis(out, -1, axis)


def _signed(img):
    """Float copy for bool/int images so differences cannot wrap around."""
    img = np.asarray(img)
    if img.dtype == np.bool_ or np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float64)
    return img


def dilate(img, radius):
    """Maximum over a (2*ry+1) x (2*rx+1) rectangle on the first two axes."""
    ry, rx = _radii(radius)
    return sliding_extremum(sliding_extremum(img, ry, 0, np.maximum), rx, 1, np.maximum)


def erode(img, radius):
    """Minimum over a (2*ry+1) x (2*rx+1) rectangle on the first two axes."""
    ry, rx = _radii(radius)
    return sliding_extremum(sliding_extremum(img, ry, 0, np.minimum), rx, 1, np.minimum)


def opening(img, radius):
    """Erosion then dilation: removes bright details smaller than the window."""
    return dilate(erode(img, radius), radius)


def closing(img, radius):
    """Dilation then erosion: fills dark details smaller than the window."""
    return erode(dilate(img, radius), radius)


def gradient(img, radius):
    """Morphological gradient (dilation - erosion), computed in a signed type."""
    work = _signed(img)
    return dilate(work, radius) - erode(work, radius)


def white_tophat(img, radius):
    """img - opening(img): bright details smaller than the window."""
    work = _signed(img)
    return work - opening(work, radius)


def black_tophat(img, radius):
    """closing(img) - img: dark details smaller than the window."""
    work = _signed(img)
    return closing(work, radius) - work
