"""Build docs/before_after.png from docs/results.json.

Every number plotted here comes from examples/run_benchmark.py, which dithers
real images and measures the result. Nothing in this file computes a metric.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "docs", "results.json")
OUT = os.path.join(ROOT, "docs", "before_after.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#76756f"
GRID = "#e4e4e0"

# Colour follows the entity, fixed order, never cycled.
COLORS = {
    "ditherfit": "#2a78d6",
    "kmeans": "#eb6834",
    "octree": "#1baf7a",
    "mediancut": "#eda100",
}
LABELS = {
    "mediancut": "median cut (Pillow default)",
    "octree": "fast octree (Pillow)",
    "kmeans": "weighted k-means",
    "ditherfit": "ditherfit",
}
DRAW_ORDER = ["mediancut", "octree", "kmeans", "ditherfit"]


def style(ax, ylabel=None, title=None, subtitle=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2, labelpad=8)
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, loc="left",
                     pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=INK_3, ha="left", va="bottom")


def rounded_column(ax, x, width, height, color, radius_frac=0.18):
    """A column with a rounded cap and a square base."""
    radius = min(width * radius_frac, height * 0.45)
    ax.add_patch(FancyBboxPatch(
        (x + radius, radius), width - 2 * radius, max(height - 2 * radius, 1e-9),
        boxstyle="round,pad={},rounding_size={}".format(radius, radius),
        linewidth=0, facecolor=color, mutation_aspect=1.0, zorder=3))
    ax.add_patch(Rectangle((x, 0), width, min(radius, height),
                           linewidth=0, facecolor=color, zorder=3))


def grouped_columns(ax, groups, series, values, fmt="{:.2f}", gap=0.02):
    """Grouped columns with a surface gap between neighbours and capped width."""
    count = len(series)
    span = 0.56
    slot = span / count
    width = slot - gap
    positions = np.arange(len(groups))
    for si, key in enumerate(series):
        for gi in range(len(groups)):
            height = values[key][gi]
            left = positions[gi] - span / 2.0 + si * slot
            rounded_column(ax, left, width, height, COLORS[key])
            ax.annotate(fmt.format(height), xy=(left + width / 2.0, height),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.6, color=INK_2, rotation=90,
                        zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=9, color=INK_2)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def panel_headline(ax, results):
    counts = "16"
    images = list(results["evaluation"])
    groups = images + ["mean"]
    values = {}
    for key in DRAW_ORDER:
        per_image = [results["evaluation"][img][counts][key]["blur_de"]
                     for img in images]
        values[key] = per_image + [float(np.mean(per_image))]
    grouped_columns(ax, groups, DRAW_ORDER, values)
    top = max(max(v) for v in values.values())
    ax.set_ylim(0, top * 1.42)
    style(ax, ylabel="perceived error (mean CIELAB dE)",
          title="Perceived colour error at 16 colours, lower is better",
          subtitle="held out images, identical Floyd-Steinberg ditherer for every palette")
    best = values["ditherfit"][-1]
    base = min(values[k][-1] for k in ("mediancut", "octree", "kmeans"))
    ax.text(0.985, 0.97,
            "mean {:.2f} dE, {:.0f} percent below the best baseline".format(
                best, 100.0 * (base - best) / base),
            transform=ax.transAxes, fontsize=10, color=INK, ha="right", va="top")


def panel_sizes(ax, results):
    counts = sorted(int(c) for c in results["aggregate"])
    ends = {}
    for key in DRAW_ORDER:
        ys = [results["aggregate"][str(c)][key]["blur_de"] for c in counts]
        ax.plot(counts, ys, color=COLORS[key], linewidth=2.0, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                solid_capstyle="round", zorder=3)
        ends[key] = ys[-1]
    spread = max(ends.values()) - min(ends.values())
    floor = 0.085 * max(ends.values()) if spread else 0.0
    placed = []
    for key, value in sorted(ends.items(), key=lambda kv: kv[1]):
        label_y = value
        for taken in placed:
            if abs(label_y - taken) < floor:
                label_y = taken + floor
        placed.append(label_y)
        ax.annotate("{:.2f}".format(value), xy=(counts[-1], label_y),
                    xytext=(9, 0), textcoords="offset points", fontsize=8.5,
                    color=INK_2, ha="left", va="center")
    ref = results["aggregate"]["32"]["kmeans"]["blur_de"]
    ax.axhline(ref, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.text(counts[0], ref, " k-means with 32 colours", fontsize=8.5,
            color=INK_3, va="bottom", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlim(counts[0] * 0.9, counts[-1] * 1.35)
    ax.set_ylim(0, max(results["aggregate"]["8"][k]["blur_de"]
                       for k in DRAW_ORDER) * 1.14)
    ax.set_xlabel("palette size", fontsize=9, color=INK_2)
    style(ax, ylabel="perceived error (mean CIELAB dE)",
          title="Sixteen fitted colours beat thirty two unfitted ones",
          subtitle="mean over the four held out images")


def panel_viewing(ax, results):
    sigmas = results["config"]["sigmas"]
    for key in DRAW_ORDER:
        ys = results["aggregate"]["16"][key]["curve"]
        ax.plot(sigmas, ys, color=COLORS[key], linewidth=2.0, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                solid_capstyle="round", zorder=3)
    base = np.array(results["aggregate"]["16"]["kmeans"]["curve"])
    fit = np.array(results["aggregate"]["16"]["ditherfit"]["curve"])
    cross = None
    for i in range(1, len(sigmas)):
        if (fit[i] - base[i]) * (fit[i - 1] - base[i - 1]) < 0:
            slope = (fit[i] - fit[i - 1]) - (base[i] - base[i - 1])
            t = (base[i - 1] - fit[i - 1]) / slope
            cross = sigmas[i - 1] + t * (sigmas[i] - sigmas[i - 1])
            break
    if cross is not None:
        ax.axvline(cross, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)),
                   zorder=2)
        ax.text(cross + 0.05, max(fit) * 0.99,
                "crossover at {:.2f} px".format(cross),
                fontsize=9, color=INK, va="top", ha="left")
    ax.set_xlabel("viewing low pass (Gaussian sigma, pixels)", fontsize=9,
                  color=INK_2)
    ax.set_xlim(min(sigmas) - 0.05, max(sigmas) + 0.08)
    ax.set_ylim(0, max(max(results["aggregate"]["16"][k]["curve"])
                       for k in DRAW_ORDER) * 1.12)
    style(ax, ylabel="mean CIELAB dE",
          title="The trade-off, stated plainly: 16 colours",
          subtitle="more per pixel dither noise bought with less average colour error")


def panel_hull(ax, results):
    counts = sorted(int(c) for c in results["aggregate"])
    groups = ["{} colours".format(c) for c in counts]
    values = {k: [100.0 * results["aggregate"][str(c)][k]["outside_hull"]
                  for c in counts] for k in DRAW_ORDER}
    grouped_columns(ax, groups, DRAW_ORDER, values, fmt="{:.0f}%")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    style(ax, ylabel="pixels outside the palette hull",
          title="Why it works: the unreachable colours shrink",
          subtitle="a pixel outside the hull is unreachable by any dither pattern")


def main():
    with open(RESULTS, encoding="utf-8") as handle:
        results = json.load(handle)
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    panel_headline(axes[0, 0], results)
    panel_sizes(axes[0, 1], results)
    panel_viewing(axes[1, 0], results)
    panel_hull(axes[1, 1], results)

    handles = [plt.Line2D([], [], marker="s", markersize=9, linestyle="none",
                          markerfacecolor=COLORS[k], markeredgecolor=COLORS[k],
                          label=LABELS[k]) for k in DRAW_ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.938),
               ncol=4, frameon=False, fontsize=9.5, labelcolor=INK_2,
               handletextpad=0.5, columnspacing=1.9)
    fig.suptitle("Palettes fitted to the ditherer that runs them",
                 fontsize=16, color=INK, x=0.052, ha="left", y=0.986)
    fig.text(0.052, 0.958,
             "Kodak reference images at 384 px long side. lam={} chosen on two "
             "separate calibration images. Perceived error is mean CIE76 dE in "
             "CIELAB after a 1 px Gaussian low pass.".format(results["lam"]),
             fontsize=9.5, color=INK_3, ha="left", va="top")
    fig.tight_layout(rect=(0.025, 0.005, 0.985, 0.905))
    fig.savefig(OUT, facecolor=SURFACE)
    print("wrote {}".format(OUT))


if __name__ == "__main__":
    main()
