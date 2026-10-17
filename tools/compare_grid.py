"""Compare two 2-D elevation grids (.npy, NaN = no data), e.g. a C++ geographic import against the Python cache.

    python tools/compare_grid.py out/import_meuse/zb_abs.npy ../SWE/data/bathy_235577_146334_600_2.npy --offset 0

Reports where both have data: max / mean absolute difference, and the no-data agreement. --offset is added to
the second grid (e.g. z_ref). Exit code 1 when the max difference exceeds --tol on the common cells.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--offset", type=float, default=0.0, help="added to b")
    ap.add_argument("--tol", type=float, default=0.05, help="max |a - b| allowed (m) on common cells")
    args = ap.parse_args()

    a = np.load(args.a).astype(np.float64)
    b = np.load(args.b).astype(np.float64) + args.offset
    if a.shape != b.shape:
        print(f"FAIL shape {a.shape} vs {b.shape}")
        return 1
    va, vb = np.isfinite(a), np.isfinite(b)
    both = va & vb
    print(f"shape {a.shape}: data a {va.mean()*100:.2f} %, b {vb.mean()*100:.2f} %, both {both.mean()*100:.2f} %, "
          f"a-only {(va & ~vb).sum()}, b-only {(vb & ~va).sum()}")
    if not both.any():
        print("FAIL no common cells")
        return 1
    d = np.abs(a[both] - b[both])
    print(f"common cells: max |diff| {d.max():.4f} m, mean {d.mean():.5f} m, median {np.median(d):.5f} m, "
          f"> {args.tol} m: {(d > args.tol).sum()} cells")
    ok = d.max() <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
