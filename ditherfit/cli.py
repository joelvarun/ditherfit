"""Command line entry point: quantize an image with a dither aware palette."""

import argparse
import json
import os

import numpy as np
from PIL import Image

from .color import linear_to_srgb, srgb_to_linear
from .dither import render
from .images import open_resized, sample_pixels, to_linear
from .metrics import mean_delta_e, outside_hull_fraction
from .palette import ditherfit_palette, pillow_palette, weighted_kmeans


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ditherfit",
        description="Quantize an image to a small palette fitted to error diffusion.")
    parser.add_argument("image", help="input image path")
    parser.add_argument("--colors", type=int, default=16, help="palette size")
    parser.add_argument("--lam", type=float, default=0.2,
                        help="dither noise penalty: 0 spans the gamut, large tends to k-means")
    parser.add_argument("--method",
                        choices=("ditherfit", "kmeans", "mediancut", "octree"),
                        default="ditherfit", help="palette builder")
    parser.add_argument("--out", default=None, help="output PNG path")
    parser.add_argument("--palette-out", default=None, help="write the palette as JSON")
    parser.add_argument("--long-side", type=int, default=0,
                        help="resize so the long side is this many pixels (0 keeps size)")
    parser.add_argument("--no-dither", action="store_true",
                        help="nearest neighbour quantization instead of error diffusion")
    parser.add_argument("--samples", type=int, default=20000,
                        help="pixels sampled for the palette fit")
    parser.add_argument("--compare", action="store_true",
                        help="also measure the median cut and k-means palettes")
    return parser


def build_palette(method, image, lin, x, weights, visibility, colors, lam):
    if method in ("mediancut", "octree"):
        return pillow_palette(image, colors, method)
    if method == "kmeans":
        return weighted_kmeans(x, weights, colors)
    return ditherfit_palette(x, weights, visibility, colors, lam=lam)[0]


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.long_side > 0:
        image = open_resized(args.image, long_side=args.long_side)
    else:
        image = Image.open(args.image).convert("RGB")
    lin = to_linear(image)
    x, weights, visibility = sample_pixels(lin, count=args.samples)

    palette = build_palette(args.method, image, lin, x, weights, visibility,
                            args.colors, args.lam)
    rec, idx = render(lin, palette, diffuse=not args.no_dither)

    out_path = args.out or "{}.{}{}.png".format(
        os.path.splitext(args.image)[0], args.method, args.colors)
    srgb = np.clip(linear_to_srgb(rec) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(srgb, mode="RGB").save(out_path)

    print("{} colours, {} palette, lam={}".format(args.colors, args.method, args.lam))
    print("  perceived dE (sigma=1 px) {:.3f}".format(mean_delta_e(lin, rec, 1.0)))
    print("  raw dE                    {:.3f}".format(mean_delta_e(lin, rec, 0.0)))
    print("  pixels outside palette hull {:.1%}".format(outside_hull_fraction(lin, palette)))
    print("  wrote {}".format(out_path))

    if args.compare:
        for method in ("mediancut", "octree", "kmeans"):
            other = build_palette(method, image, lin, x, weights, visibility,
                                  args.colors, args.lam)
            other_rec, _ = render(lin, other, diffuse=not args.no_dither)
            print("  {:<10} perceived dE {:.3f}  raw dE {:.3f}  outside hull {:.1%}".format(
                method, mean_delta_e(lin, other_rec, 1.0),
                mean_delta_e(lin, other_rec, 0.0), outside_hull_fraction(lin, other)))

    if args.palette_out:
        srgb_palette = np.clip(linear_to_srgb(palette) * 255.0 + 0.5, 0, 255).astype(int)
        with open(args.palette_out, "w", encoding="utf-8") as handle:
            json.dump({
                "colors": args.colors,
                "method": args.method,
                "lam": args.lam,
                "srgb": srgb_palette.tolist(),
                "hex": ["#{:02x}{:02x}{:02x}".format(*row) for row in srgb_palette],
                "linear": np.asarray(palette).round(6).tolist(),
            }, handle, indent=2)
        print("  wrote {}".format(args.palette_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
