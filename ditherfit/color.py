"""sRGB, linear light, CIE XYZ and CIELAB conversions.

Error diffusion averages light, so every mixing calculation in this package
runs in linear RGB. Error is reported in CIELAB, which is where a difference
of one unit is roughly one just noticeable step.
"""

import numpy as np

# sRGB primaries, D65 white point (IEC 61966-2-1).
M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_linear(u):
    """Decode gamma encoded sRGB in [0, 1] to linear light in [0, 1]."""
    u = np.asarray(u, dtype=np.float64)
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Encode linear light in [0, 1] back to gamma encoded sRGB in [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def linear_to_lab(lin):
    """Convert linear RGB (..., 3) to CIELAB (..., 3)."""
    lin = np.asarray(lin, dtype=np.float64)
    shape = lin.shape
    xyz = lin.reshape(-1, 3) @ M_RGB2XYZ.T
    t = xyz / WHITE_D65
    f = np.where(t > _EPS, np.cbrt(np.maximum(t, 0.0)), (_KAPPA * t + 16.0) / 116.0)
    lab = np.stack([
        116.0 * f[:, 1] - 16.0,
        500.0 * (f[:, 0] - f[:, 1]),
        200.0 * (f[:, 1] - f[:, 2]),
    ], axis=1)
    return lab.reshape(shape)


def delta_e76(lab_a, lab_b):
    """Per pixel CIE76 colour difference between two CIELAB arrays."""
    d = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return np.sqrt((d ** 2).sum(axis=-1))


def luminance(lin):
    """Relative luminance Y of linear RGB values."""
    return np.asarray(lin, dtype=np.float64).reshape(-1, 3) @ M_RGB2XYZ[1]


def lightness_weight(lin, floor=1e-4):
    """Per pixel accuracy weight approximating dL*/dY, which scales as Y^(-2/3).

    A linear RGB error of a given size costs more CIELAB units in the shadows
    than in the highlights. The palette fit is a least squares problem in
    linear RGB, so this weight is what keeps it from spending all of its
    precision on bright pixels.
    """
    y = np.clip(luminance(lin), floor, None) ** (-2.0 / 3.0)
    return (y / y.mean()).reshape(np.asarray(lin).shape[:-1])
