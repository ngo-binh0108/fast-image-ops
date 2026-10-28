"""
fast_image_ops: image filters whose cost per pixel does not grow with the window size.
"""
from .box import box_mean, box_sum
from .guided import guided_filter, fast_guided_filter, resize_bilinear
from .morphology import dilate, erode, opening, closing, gradient, white_tophat, black_tophat
from .distance import distance_transform, squared_distance_1d

__version__ = "0.1.0"

__all__ = [
    "box_mean", "box_sum",
    "guided_filter", "fast_guided_filter", "resize_bilinear",
    "dilate", "erode", "opening", "closing", "gradient", "white_tophat", "black_tophat",
    "distance_transform", "squared_distance_1d",
]
