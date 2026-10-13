"""Floyd-Steinberg error diffusion in linear light.

The diffusion loop is strictly sequential, so the inner loop works on Python
lists of floats and resolves the nearest palette entry through a precomputed
cubic lookup table instead of a distance computation per pixel.
"""

import numpy as np

FS_WEIGHTS = (7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0)


def nearest_lut(palette, bits=6):
    """Nearest palette index for every cell of a 2**bits cube in linear RGB."""
    palette = np.asarray(palette, dtype=np.float64)
    n = 1 << bits
    axis = (np.arange(n) + 0.5) / n
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    out = np.empty(len(grid), dtype=np.int32)
    chunk = 8192
    for start in range(0, len(grid), chunk):
        block = grid[start:start + chunk]
        d = ((block[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk] = d.argmin(axis=1)
    return out


def nearest_indices(lin, palette, lut=None, bits=6):
    """Plain nearest neighbour quantization, no diffusion."""
    lin = np.asarray(lin, dtype=np.float64)
    if lut is None:
        lut = nearest_lut(palette, bits)
    n = 1 << bits
    q = np.clip(lin, 0.0, 1.0)
    idx = (q * (n - 1)).astype(np.int32)
    flat = (idx[..., 0] * n + idx[..., 1]) * n + idx[..., 2]
    return lut[flat].astype(np.int16)


def dither_indices(lin, palette, lut=None, bits=6, serpentine=True):
    """Serpentine Floyd-Steinberg diffusion, returns an index image."""
    lin = np.asarray(lin, dtype=np.float64)
    palette = np.asarray(palette, dtype=np.float64)
    if lut is None:
        lut = nearest_lut(palette, bits)
    height, width, _ = lin.shape
    n = 1 << bits
    scale = n - 1
    table = lut.tolist()
    pal = palette.tolist()
    rows = lin.tolist()
    out = [[0] * width for _ in range(height)]
    w_right, w_down_left, w_down, w_down_right = FS_WEIGHTS
    for y in range(height):
        row = rows[y]
        below = rows[y + 1] if y + 1 < height else None
        reverse = serpentine and (y & 1) == 1
        step = -1 if reverse else 1
        order = range(width - 1, -1, -1) if reverse else range(width)
        out_row = out[y]
        for x in order:
            px = row[x]
            r, g, b = px[0], px[1], px[2]
            cr = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
            cg = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
            cb = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
            k = table[(int(cr * scale) * n + int(cg * scale)) * n + int(cb * scale)]
            out_row[x] = k
            entry = pal[k]
            er = r - entry[0]
            eg = g - entry[1]
            eb = b - entry[2]
            ahead = x + step
            if 0 <= ahead < width:
                t = row[ahead]
                t[0] += er * w_right
                t[1] += eg * w_right
                t[2] += eb * w_right
            if below is not None:
                behind = x - step
                if 0 <= behind < width:
                    t = below[behind]
                    t[0] += er * w_down_left
                    t[1] += eg * w_down_left
                    t[2] += eb * w_down_left
                t = below[x]
                t[0] += er * w_down
                t[1] += eg * w_down
                t[2] += eb * w_down
                if 0 <= ahead < width:
                    t = below[ahead]
                    t[0] += er * w_down_right
                    t[1] += eg * w_down_right
                    t[2] += eb * w_down_right
    return np.array(out, dtype=np.int16)


def render(lin, palette, diffuse=True, bits=6):
    """Quantize an image and return (linear reconstruction, index image)."""
    palette = np.asarray(palette, dtype=np.float64)
    lut = nearest_lut(palette, bits)
    if diffuse:
        idx = dither_indices(lin, palette, lut=lut, bits=bits)
    else:
        idx = nearest_indices(lin, palette, lut=lut, bits=bits)
    return palette[idx], idx
