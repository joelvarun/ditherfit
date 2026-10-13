"""Palette construction: nearest neighbour baselines and the ditherfit fit.

Classical palette builders (median cut, Wu, k-means) minimise the distance
from each pixel to its single closest palette entry. Error diffusion does not
work that way: it reproduces a local average, so the colour a dithered region
can show is any point in the convex hull of the palette entries used nearby.
A palette that is optimal under the nearest neighbour assumption therefore
collapses onto the dense part of the colour histogram and leaves the saturated
corners of the image gamut outside its hull, where no amount of diffusion can
reach them.

ditherfit optimises the palette for the ditherer that will actually run it:

    minimise  sum_p  w_p * || c_p - sum_j a_pj q_j ||^2
                   + lam * w_p * v_p * sum_j a_pj || q_j - c_p ||^2

subject to a_p in the probability simplex over the m palette entries nearest
to pixel c_p. The first term is reproduction error against the mixture that
diffusion can realise. The second term is the variance of that mixture, which
is the amplitude of the visible dither noise, scaled by a per pixel visibility
weight v_p that is near one in flat regions and small in busy texture.

lam interpolates between two known regimes. At lam = 0 the fit is pure hull
coverage: the palette spreads to span the gamut and the output is accurate on
average but grainy. As lam grows the mixtures collapse towards single entries
and the solution approaches weighted k-means. Both steps of the alternation
are exact, so the objective decreases monotonically up to the gamut clip.
"""

import numpy as np

from .color import srgb_to_linear


def simplex_project(v):
    """Project every row of v onto the probability simplex."""
    n = v.shape[1]
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, n + 1)
    cond = (u - css / ind) > 0
    rho = n - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(len(v)), rho] / (rho + 1.0)
    return np.maximum(v - theta[:, None], 0.0)


def _pairwise_sq(x, c):
    return ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)


def weighted_kmeans(x, weights, n_colors, iters=30, seed=0, tol=1e-7):
    """Weighted k-means in linear RGB, the nearest neighbour optimal baseline."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    probs = weights / weights.sum()
    centres = x[rng.choice(len(x), size=n_colors, replace=False, p=probs)].copy()
    prev = np.inf
    for _ in range(iters):
        d = _pairwise_sq(x, centres)
        assign = d.argmin(axis=1)
        cost = float((weights * d.min(axis=1)).sum())
        for j in range(n_colors):
            mask = assign == j
            if not mask.any():
                centres[j] = x[(weights * d.min(axis=1)).argmax()]
                continue
            w = weights[mask][:, None]
            centres[j] = (x[mask] * w).sum(axis=0) / w.sum()
        if abs(prev - cost) <= tol * max(cost, 1e-12):
            break
        prev = cost
    return np.clip(centres, 0.0, 1.0)


PILLOW_METHODS = {
    "mediancut": 0,   # Image.Quantize.MEDIANCUT, the default for RGB images
    "maxcoverage": 1,  # Image.Quantize.MAXCOVERAGE
    "octree": 2,      # Image.Quantize.FASTOCTREE
}


def pillow_palette(image, n_colors, method="mediancut"):
    """A palette from Pillow's own quantizers, converted to linear RGB.

    These are the palettes tools ship today: Image.quantize defaults to median
    cut for RGB input, and the fast octree quantizer is the usual alternative.
    Both minimise nearest neighbour error and know nothing about the ditherer.
    """
    quantized = image.convert("RGB").quantize(
        colors=n_colors,
        method=PILLOW_METHODS[method],
        dither=0,  # Image.Dither.NONE, the palette is what matters here
    )
    flat = quantized.getpalette()[: 3 * n_colors]
    pal = np.array(flat, dtype=np.float64).reshape(-1, 3) / 255.0
    return srgb_to_linear(pal)


def mediancut_palette(image, n_colors):
    """Shorthand for Pillow's median cut palette."""
    return pillow_palette(image, n_colors, method="mediancut")


def mixing_weights(x, palette, visibility, lam, m=4, pg_iters=40):
    """Solve the per pixel simplex problem, returning (weights, neighbours)."""
    x = np.asarray(x, dtype=np.float64)
    palette = np.asarray(palette, dtype=np.float64)
    m = min(m, len(palette))
    d = _pairwise_sq(x, palette)
    near = np.argsort(d, axis=1)[:, :m]
    cand = palette[near]
    dist = np.take_along_axis(d, near, axis=1)
    gram = np.einsum("nmd,nkd->nmk", cand, cand)
    proj = np.einsum("nmd,nd->nm", cand, x)
    linear = -2.0 * proj + lam * visibility[:, None] * dist
    a = np.full((len(x), m), 1.0 / m)
    step = 1.0 / (2.0 * np.abs(gram).sum(axis=2).max(axis=1) + 1e-9)
    for _ in range(pg_iters):
        grad = 2.0 * np.einsum("nmk,nk->nm", gram, a) + linear
        a = simplex_project(a - step[:, None] * grad)
    return a, near


def ditherfit_palette(x, weights, visibility, n_colors, lam=0.3, iters=18,
                      m=4, pg_iters=40, seed=0, init=None):
    """Alternating fit of a palette to the mixtures error diffusion can reach."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=np.float64)
    palette = (weighted_kmeans(x, weights, n_colors, seed=seed)
               if init is None else np.array(init, dtype=np.float64))
    eye = np.arange(n_colors)
    history = []
    for _ in range(iters):
        a, near = mixing_weights(x, palette, visibility, lam, m=m, pg_iters=pg_iters)
        full = np.zeros((len(x), n_colors))
        np.put_along_axis(full, near, a, axis=1)
        mix = full @ palette
        resid = ((x - mix) ** 2).sum(axis=1)
        history.append(float((weights * resid).sum() / weights.sum()))

        wa = full * weights[:, None]
        noise = full * (weights * visibility)[:, None]
        lhs = wa.T @ full
        rhs = wa.T @ x + lam * (noise.T @ x)
        lhs[eye, eye] += lam * noise.sum(axis=0) + 1e-9
        palette = np.clip(np.linalg.solve(lhs, rhs), 0.0, 1.0)

        dead = full.sum(axis=0) < 1e-6
        if dead.any():
            worst = np.argsort(-weights * resid)[: int(dead.sum())]
            palette[dead] = x[worst]
    return palette, history
