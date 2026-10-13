"""Sanity tests: run with `python -m pytest tests` or `python tests/test_core.py`."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ditherfit.color import linear_to_lab, linear_to_srgb, srgb_to_linear
from ditherfit.dither import render
from ditherfit.metrics import mean_delta_e, outside_hull_fraction
from ditherfit.palette import (ditherfit_palette, mixing_weights,
                               simplex_project, weighted_kmeans)


def test_srgb_round_trip():
    u = np.linspace(0.0, 1.0, 64)
    assert np.allclose(linear_to_srgb(srgb_to_linear(u)), u, atol=1e-9)


def test_lab_anchors():
    white = linear_to_lab(np.array([[1.0, 1.0, 1.0]]))[0]
    black = linear_to_lab(np.array([[0.0, 0.0, 0.0]]))[0]
    assert np.allclose(white, [100.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(black, [0.0, 0.0, 0.0], atol=1e-6)


def test_simplex_projection():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 4)) * 3.0
    a = simplex_project(v)
    assert (a >= -1e-12).all()
    assert np.allclose(a.sum(axis=1), 1.0, atol=1e-9)


def test_mixing_weights_hit_interior_colours():
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = np.array([[0.25, 0.25, 0.25]])
    a, near = mixing_weights(x, palette, np.zeros(1), lam=0.0)
    mix = (a[:, :, None] * palette[near]).sum(axis=1)
    assert np.allclose(mix, x, atol=1e-3)


def test_large_lambda_approaches_kmeans():
    rng = np.random.default_rng(1)
    x = np.clip(rng.normal(0.5, 0.18, size=(2000, 3)), 0.0, 1.0)
    w = np.ones(len(x))
    v = np.ones(len(x))
    km = weighted_kmeans(x, w, 6, seed=3)
    tight = ditherfit_palette(x, w, v, 6, lam=40.0, iters=12, seed=3)[0]
    loose = ditherfit_palette(x, w, v, 6, lam=0.0, iters=12, seed=3)[0]
    spread = lambda p: float(np.abs(p - p.mean(axis=0)).mean())
    assert spread(loose) > spread(tight)
    assert spread(tight) <= spread(km) * 1.25


def test_objective_decreases():
    rng = np.random.default_rng(2)
    x = np.clip(rng.normal(0.5, 0.2, size=(1500, 3)), 0.0, 1.0)
    _, history = ditherfit_palette(x, np.ones(len(x)), np.ones(len(x)),
                                   8, lam=0.2, iters=10, seed=0)
    assert history[-1] <= history[0] + 1e-12


def test_dither_beats_nearest_on_a_gradient():
    ramp = np.linspace(0.0, 1.0, 96)
    lin = np.repeat(ramp[None, :, None], 48, axis=0).repeat(3, axis=2)
    palette = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    diffused, idx = render(lin, palette, diffuse=True)
    plain, _ = render(lin, palette, diffuse=False)
    assert idx.min() >= 0 and idx.max() < len(palette)
    assert mean_delta_e(lin, diffused, 1.0) < mean_delta_e(lin, plain, 1.0)


def test_outside_hull_fraction_bounds():
    lin = np.full((8, 8, 3), 0.5)
    inside = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    outside = inside * 0.1
    assert outside_hull_fraction(lin, inside) == 0.0
    assert outside_hull_fraction(lin, outside) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok {}".format(name))
