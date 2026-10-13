"""Error measures for dithered output.

Measuring a dithered image pixel by pixel scores the dither pattern itself,
not the colour a viewer sees. The eye integrates over a small spatial window,
so the headline metric here low passes both the original and the reproduction
with a small Gaussian before converting to CIELAB. Raw (unblurred) error is
reported alongside it because it is the part of the trade-off that gets worse
when a palette spreads out, and hiding that would make the comparison dishonest.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

from .color import delta_e76, linear_to_lab

DEFAULT_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def mean_delta_e(lin, rec, sigma=1.0):
    """Mean CIE76 error after a sigma pixel Gaussian low pass, in CIELAB."""
    lin = np.asarray(lin, dtype=np.float64)
    rec = np.asarray(rec, dtype=np.float64)
    if sigma > 0:
        lin = gaussian_filter(lin, (sigma, sigma, 0))
        rec = gaussian_filter(rec, (sigma, sigma, 0))
    return float(delta_e76(linear_to_lab(lin), linear_to_lab(rec)).mean())


def error_curve(lin, rec, sigmas=DEFAULT_SIGMAS):
    """Mean error as a function of the viewing low pass width."""
    return [mean_delta_e(lin, rec, sigma=s) for s in sigmas]


def outside_hull_fraction(lin, palette):
    """Share of pixels whose colour lies outside the palette's convex hull.

    These are the pixels error diffusion can never reproduce on average: the
    residual has a fixed sign and is pushed into the neighbours forever.
    """
    pts = np.asarray(palette, dtype=np.float64)
    if len(pts) < 4:
        return 1.0
    try:
        hull = Delaunay(pts)
    except Exception:
        return float("nan")
    flat = np.asarray(lin, dtype=np.float64).reshape(-1, 3)
    return float((hull.find_simplex(flat) < 0).mean())


def noise_visibility(lin, eps=1e-6):
    """Per pixel weight for how visible dither noise is, from local contrast.

    Flat sky shows granularity; busy foliage masks it. The map is lightness
    gradient magnitude normalised by its own median, which keeps the single
    lam knob comparable across images.
    """
    lightness = linear_to_lab(np.asarray(lin, dtype=np.float64))[:, :, 0]
    gy, gx = np.gradient(lightness)
    grad = np.hypot(gx, gy)
    tau = float(np.median(grad)) + eps
    return 1.0 / (1.0 + (grad / tau) ** 2)
