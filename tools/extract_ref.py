"""Reference extraction of a GeoTIFF window, block-averaged like cases/geotiff.py, for validating the C++ import.

    python tools/extract_ref.py <geotiff> xmin,ymin,xmax,ymax <cell> out.npy

The box must be aligned on the raster pixel grid and the cell must be a multiple of the pixel size. Output is
(nx, ny) float32 in the solver layout (i east, j north), NaN where fewer than 50 % of the pixels are valid.
"""
import sys

import numpy as np
import rasterio
from rasterio.windows import Window


def main():
    path, box, cell, out = sys.argv[1], [float(v) for v in sys.argv[2].split(",")], float(sys.argv[3]), sys.argv[4]
    xmin, ymin, xmax, ymax = box
    with rasterio.open(path) as src:
        res = src.res[0]
        t = src.transform
        col0 = (xmin - t.c) / t.a
        row0 = (ymax - t.f) / t.e
        if abs(col0 - round(col0)) > 1e-6 or abs(row0 - round(row0)) > 1e-6:
            raise SystemExit(f"box not aligned on the pixel grid (col {col0}, row {row0})")
        f = int(round(cell / res))
        n = int(round((xmax - xmin) / cell))
        m = int(round((ymax - ymin) / cell))
        a = src.read(1, window=Window(int(round(col0)), int(round(row0)), n * f, m * f)).astype(np.float32)
        valid = np.isfinite(a) & (a != src.nodata)
    count = valid.reshape(m, f, n, f).sum(axis=(1, 3))
    total = np.where(valid, a, 0.0).reshape(m, f, n, f).sum(axis=(1, 3))
    z = np.where(count >= f * f / 2, total / np.maximum(count, 1), np.nan).astype(np.float32)
    z = np.ascontiguousarray(z[::-1].T)  # rows north->south -> [i east, j north]
    np.save(out, z)
    print(f"{out}: {z.shape}, data {np.isfinite(z).mean()*100:.2f} %")


if __name__ == "__main__":
    main()
