"""Benchmark harness: calibrate lam, then score held out images.

lam is chosen on two calibration images and then applied unchanged to four
evaluation images, so the headline number is not tuned on the images it is
reported for.
"""

import time

import numpy as np

from .dither import render
from .images import (CALIBRATION, EVALUATION, fetch_kodak, open_resized,
                     sample_pixels, to_linear)
from .metrics import DEFAULT_SIGMAS, error_curve, mean_delta_e, outside_hull_fraction
from .palette import ditherfit_palette, pillow_palette, weighted_kmeans

LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
COLOR_COUNTS = (8, 16, 32)
METHOD_LABELS = {
    "mediancut": "median cut (Pillow)",
    "octree": "fast octree (Pillow)",
    "kmeans": "weighted k-means",
    "ditherfit": "ditherfit",
}
BASELINE = "kmeans"  # strongest nearest neighbour palette, the headline "before"


def load_case(number, long_side=384, samples=20000, seed=0):
    """Prepare one image: linear pixels plus the sampled fitting problem."""
    image = open_resized(fetch_kodak(number), long_side=long_side)
    lin = to_linear(image)
    x, weights, visibility = sample_pixels(lin, count=samples, seed=seed)
    return {
        "name": "kodim{:02d}".format(number),
        "image": image,
        "lin": lin,
        "x": x,
        "weights": weights,
        "visibility": visibility,
    }


def score(lin, palette, sigmas=DEFAULT_SIGMAS):
    """Dither with `palette` and measure the result."""
    rec, idx = render(lin, palette, diffuse=True)
    plain, _ = render(lin, palette, diffuse=False)
    return {
        "blur_de": mean_delta_e(lin, rec, sigma=1.0),
        "raw_de": mean_delta_e(lin, rec, sigma=0.0),
        "curve": error_curve(lin, rec, sigmas=sigmas),
        "nodither_de": mean_delta_e(lin, plain, sigma=1.0),
        "outside_hull": outside_hull_fraction(lin, palette),
        "palette": np.asarray(palette).round(6).tolist(),
    }, rec, idx


def calibrate(n_colors=16, lambdas=LAMBDAS, numbers=CALIBRATION, long_side=384):
    """Sweep lam on the calibration images and pick the best perceived error."""
    cases = [load_case(n, long_side=long_side) for n in numbers]
    rows = []
    for lam in lambdas:
        blur, raw = [], []
        for case in cases:
            palette, _ = ditherfit_palette(
                case["x"], case["weights"], case["visibility"], n_colors, lam=lam)
            rec, _ = render(case["lin"], palette, diffuse=True)
            blur.append(mean_delta_e(case["lin"], rec, sigma=1.0))
            raw.append(mean_delta_e(case["lin"], rec, sigma=0.0))
        rows.append({
            "lam": lam,
            "blur_de": float(np.mean(blur)),
            "raw_de": float(np.mean(raw)),
            "per_image": {c["name"]: b for c, b in zip(cases, blur)},
        })
        print("  lam={:<5} perceived dE={:.3f}  raw dE={:.3f}".format(
            lam, rows[-1]["blur_de"], rows[-1]["raw_de"]), flush=True)
    best = min(rows, key=lambda r: r["blur_de"])
    return {
        "n_colors": n_colors,
        "images": [c["name"] for c in cases],
        "sweep": rows,
        "chosen_lam": best["lam"],
    }


def evaluate(lam, numbers=EVALUATION, color_counts=COLOR_COUNTS, long_side=384):
    """Score median cut, k-means and ditherfit on the evaluation images."""
    results = {}
    for number in numbers:
        case = load_case(number, long_side=long_side)
        per_count = {}
        for n_colors in color_counts:
            palettes = {
                "mediancut": pillow_palette(case["image"], n_colors, "mediancut"),
                "octree": pillow_palette(case["image"], n_colors, "octree"),
                "kmeans": weighted_kmeans(case["x"], case["weights"], n_colors),
            }
            palettes["ditherfit"] = ditherfit_palette(
                case["x"], case["weights"], case["visibility"], n_colors, lam=lam)[0]
            entry = {}
            for method, palette in palettes.items():
                metrics, _, _ = score(case["lin"], palette)
                entry[method] = metrics
                print("  {} P={:<3} {:<20} perceived dE={:.3f} raw dE={:.3f} "
                      "outside hull={:.1%}".format(
                          case["name"], n_colors, METHOD_LABELS[method],
                          metrics["blur_de"], metrics["raw_de"],
                          metrics["outside_hull"]), flush=True)
            per_count[str(n_colors)] = entry
        results[case["name"]] = per_count
    return results


def aggregate(evaluation, color_counts=COLOR_COUNTS):
    """Mean metrics over the evaluation images, per palette size and method."""
    out = {}
    for n_colors in color_counts:
        key = str(n_colors)
        block = {}
        for method in METHOD_LABELS:
            block[method] = {
                field: float(np.mean([evaluation[img][key][method][field]
                                      for img in evaluation]))
                for field in ("blur_de", "raw_de", "nodither_de", "outside_hull")
            }
            block[method]["curve"] = np.mean(
                [evaluation[img][key][method]["curve"] for img in evaluation],
                axis=0).tolist()
        out[key] = block
    return out


def run(long_side=384, n_colors_cal=16):
    """Full benchmark. Returns a JSON serialisable results dictionary."""
    started = time.time()
    print("calibrating lam on {}".format(list(CALIBRATION)), flush=True)
    cal = calibrate(n_colors=n_colors_cal, long_side=long_side)
    lam = cal["chosen_lam"]
    print("chosen lam = {}".format(lam), flush=True)
    print("evaluating on {}".format(list(EVALUATION)), flush=True)
    ev = evaluate(lam, long_side=long_side)
    return {
        "config": {
            "long_side": long_side,
            "sigmas": list(DEFAULT_SIGMAS),
            "color_counts": list(COLOR_COUNTS),
            "lambdas": list(LAMBDAS),
            "calibration_images": ["kodim{:02d}".format(n) for n in CALIBRATION],
            "evaluation_images": ["kodim{:02d}".format(n) for n in EVALUATION],
            "metric": "mean CIE76 dE in CIELAB, sigma=1 px Gaussian low pass",
        },
        "calibration": cal,
        "lam": lam,
        "evaluation": ev,
        "aggregate": aggregate(ev),
        "wall_seconds": round(time.time() - started, 1),
    }
