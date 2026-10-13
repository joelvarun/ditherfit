"""Smallest useful demo: one image, three palettes, printed numbers.

    python examples/demo_single.py [kodak_number] [palette_size]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ditherfit.benchmark import METHOD_LABELS, load_case, score  # noqa: E402
from ditherfit.palette import ditherfit_palette, pillow_palette, weighted_kmeans  # noqa: E402


def main():
    number = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    colors = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    case = load_case(number)
    palettes = {
        "mediancut": pillow_palette(case["image"], colors, "mediancut"),
        "octree": pillow_palette(case["image"], colors, "octree"),
        "kmeans": weighted_kmeans(case["x"], case["weights"], colors),
        "ditherfit": ditherfit_palette(case["x"], case["weights"],
                                       case["visibility"], colors, lam=0.2)[0],
    }
    print("kodim{:02d}, {} colours, {}x{} px".format(
        number, colors, case["lin"].shape[1], case["lin"].shape[0]))
    print("{:<22} {:>12} {:>9} {:>14} {:>14}".format(
        "palette", "perceived dE", "raw dE", "outside hull", "no dither dE"))
    for method, palette in palettes.items():
        metrics, _, _ = score(case["lin"], palette)
        print("{:<22} {:>12.3f} {:>9.3f} {:>13.1%} {:>14.3f}".format(
            METHOD_LABELS[method], metrics["blur_de"], metrics["raw_de"],
            metrics["outside_hull"], metrics["nodither_de"]))


if __name__ == "__main__":
    main()
