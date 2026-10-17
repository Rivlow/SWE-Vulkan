"""Well-balancedness check: still water over the dam_break bump must stay at rest.

    python tools/lake_at_rest.py [--steps 1000] [--scheme ssprk2]

Runs `swe --headless --case lake_at_rest`, then checks on wet cells that the free surface stays at eta = 0.5
and that the momentum stays ~0. Exit code 1 on failure.

Reference (Python/Taichi, same setup via `dump_reference.py --still-water 0.5`, ssprk2, 1000 steps):
max|eta - 0.5| = 9.7e-7 and max|hu|,|hv| = 1.1e-5 in deep water - f32 round-off of the hydrostatic balance,
not a scheme defect. Tolerances are set a few times above that.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--scheme", default="ssprk2")
    ap.add_argument("--preset", default="debug")
    ap.add_argument("--eta-tol", type=float, default=1e-5)
    ap.add_argument("--mom-tol", type=float, default=5e-5)
    args = ap.parse_args()

    exe = ROOT / "build" / args.preset / "bin" / "swe.exe"
    out = ROOT / "out" / f"cpp_lake_at_rest_{args.scheme}_{args.steps}"
    r = subprocess.run([str(exe), "--headless", "--no-validation", "--case", "lake_at_rest", "--scheme", args.scheme,
                        "--steps", str(args.steps), "--dump", str(out)], cwd=ROOT)
    if r.returncode != 0:
        return 1

    q = np.load(out / "Q.npy").astype(np.float64)
    zb = np.load(out / "zb.npy").astype(np.float64)
    h, hu, hv = q[..., 0], q[..., 1], q[..., 2]
    wet = h > 1e-3
    eta_err = float(np.abs(h + zb - 0.5)[wet].max())
    mom = float(max(np.abs(hu).max(), np.abs(hv).max()))
    dry_water = float(h[~wet].max()) if (~wet).any() else 0.0
    print(f"lake at rest, {args.scheme}, {args.steps} steps: wet cells {int(wet.sum())}, "
          f"max|eta - 0.5| = {eta_err:.3e}, max|hu|,|hv| = {mom:.3e}, max h on dry cells = {dry_water:.3e}")
    ok = eta_err <= args.eta_tol and mom <= args.mom_tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
