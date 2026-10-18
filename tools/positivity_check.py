"""Drain limiter check on a headless dump: python tools/positivity_check.py out/dir [--conserve] [--tol 1e-5]

PASS when every h >= 0, no cell was clamped to 0 by the solver (meta.json: clamp_cells) and, with --conserve
(walls everywhere, no zones), the volume matches the initial one to --tol (relative; round-off of f32 cell updates).
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dir")
ap.add_argument("--conserve", action="store_true", help="also require volume == initial volume")
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--expect-clamps", action="store_true",
                help="sanity check of a run with the limiter off: PASS when cells were clamped (the case does need the limiter)")
args = ap.parse_args()

q = np.load(f"{args.dir}/Q.npy")
m = json.load(open(f"{args.dir}/meta.json"))
h = q[..., 0].astype(np.float64)

ok = True
hmin = float(h.min())
print(f"{args.dir}: t = {m['t']:.4f} s, {m['steps']} steps, drain_passes = {m.get('drain_passes')}, "
      f"limited cells (last step) = {m.get('limited_cells')}")
print(f"  h min = {hmin:.3e} m, negative cells = {(h < 0).sum()}")
ok &= hmin >= 0.0
clamp_n = m.get("clamp_cells", 0)
print(f"  clamped cells = {clamp_n}, largest clamp = {m.get('clamp_max', 0):.3e} m")
if args.expect_clamps:
    print("PASS (limiter off: the case does trigger clamps)" if clamp_n > 0 else "FAIL (limiter off: no clamp, the case is too easy)")
    sys.exit(0 if clamp_n > 0 else 1)
ok &= clamp_n == 0
if args.conserve and "volume0" in m:
    v0, v = m["volume0"], m["volume"]
    rel = abs(v - v0) / v0 if v0 > 0 else abs(v - v0)
    print(f"  volume = {v:.6f} m3 (initial {v0:.6f}), relative change = {rel:.3e}")
    ok &= rel <= args.tol
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
