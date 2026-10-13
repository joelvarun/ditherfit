"""Run the full benchmark and write docs/results.json.

Downloads the six Kodak reference images on first run (about 4 MB) and takes
a couple of minutes on a laptop CPU.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ditherfit.benchmark import run  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "results.json")


def main():
    results = run()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    agg = results["aggregate"]["16"]
    print()
    print("16 colour palettes, mean over the four evaluation images")
    for method in ("mediancut", "octree", "kmeans", "ditherfit"):
        print("  {:<18} perceived dE {:.2f}   raw dE {:.2f}   outside hull {:.1%}".format(
            method, agg[method]["blur_de"], agg[method]["raw_de"],
            agg[method]["outside_hull"]))
    print("wrote {} in {} s".format(OUT, results["wall_seconds"]))


if __name__ == "__main__":
    main()
