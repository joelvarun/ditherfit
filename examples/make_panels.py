"""Render the visual comparison: docs/panels.png.

Top row is the full 16 colour reproduction from each palette builder, bottom
row is a 3x crop of the same region so the dither texture is visible.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ditherfit.benchmark import load_case  # noqa: E402
from ditherfit.color import linear_to_srgb  # noqa: E402
from ditherfit.dither import render  # noqa: E402
from ditherfit.metrics import mean_delta_e  # noqa: E402
from ditherfit.palette import ditherfit_palette, pillow_palette, weighted_kmeans  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "panels.png")
CROP = (120, 150, 230, 230)  # left, top, right, bottom in the 384 px wide image
INK = "#1b1b1f"


def show(ax, lin, title, subtitle=None):
    ax.imshow(np.clip(linear_to_srgb(lin), 0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#d4d4d8")
    ax.set_title(title, fontsize=10, color=INK, pad=6)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9, color="#52525b", labelpad=4)


def main(number=23, colors=16, lam=0.2):
    case = load_case(number)
    lin = case["lin"]
    palettes = [
        ("original", None),
        ("median cut (Pillow)", pillow_palette(case["image"], colors, "mediancut")),
        ("fast octree (Pillow)", pillow_palette(case["image"], colors, "octree")),
        ("weighted k-means", weighted_kmeans(case["x"], case["weights"], colors)),
        ("ditherfit", ditherfit_palette(case["x"], case["weights"],
                                        case["visibility"], colors, lam=lam)[0]),
    ]
    left, top, right, bottom = CROP
    fig, axes = plt.subplots(2, 5, figsize=(16.5, 6.0), dpi=150)
    fig.patch.set_facecolor("white")
    for col, (name, palette) in enumerate(palettes):
        if palette is None:
            img = lin
            subtitle = "24 bit reference"
        else:
            img, _ = render(lin, palette, diffuse=True)
            subtitle = "perceived dE {:.2f}".format(mean_delta_e(lin, img, 1.0))
        show(axes[0, col], img, name, subtitle)
        show(axes[1, col], img[top:bottom, left:right], "")
    axes[1, 0].set_ylabel("3x crop", fontsize=9, color="#52525b")
    fig.suptitle("{} colours, Floyd-Steinberg diffusion in linear light, kodim{:02d}".format(
        colors, number), fontsize=12, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, facecolor="white")
    print("wrote {}".format(OUT))


if __name__ == "__main__":
    main()
