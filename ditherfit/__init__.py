"""ditherfit: build colour palettes for the ditherer that will run them."""

from .color import delta_e76, linear_to_lab, linear_to_srgb, srgb_to_linear
from .dither import dither_indices, nearest_indices, render
from .metrics import error_curve, mean_delta_e, noise_visibility, outside_hull_fraction
from .palette import (ditherfit_palette, mediancut_palette, mixing_weights,
                      pillow_palette, weighted_kmeans)

__version__ = "0.1.0"

__all__ = [
    "delta_e76",
    "dither_indices",
    "ditherfit_palette",
    "error_curve",
    "linear_to_lab",
    "linear_to_srgb",
    "mean_delta_e",
    "mediancut_palette",
    "mixing_weights",
    "nearest_indices",
    "noise_visibility",
    "outside_hull_fraction",
    "pillow_palette",
    "render",
    "srgb_to_linear",
    "weighted_kmeans",
]
