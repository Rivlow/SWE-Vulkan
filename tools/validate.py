"""Validate the C++/Vulkan solver against the Python/Taichi reference.

    python tools/validate.py dam_break ssprk2 200            # one run
    python tools/validate.py --all                            # every scheme on dam_break + lake at rest

For each run: tools/dump_reference.py (cached in out/ref_*), swe --headless --dump, tools/compare.py.
Tolerances: relative L1 <= 1e-5 and at most max(--front-cells, 0.05 % of the cells) above 1e-5 (round-off is
amplified at shocks and wet/dry fronts: Taichi CPU vs Taichi Vulkan shows the same pattern, see README).
The bed elevation zb must match exactly (same numpy preprocessing ported to C++).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCHEMES = ["euler", "ssprk2", "ssprk3", "rk4", "dopri5"]


def find_exe(preset):
    exe = ROOT / "build" / preset / "bin" / "swe.exe"
    if not exe.exists():
        exe = exe.with_suffix("")
    if not exe.exists():
        raise SystemExit(f"swe executable not found in build/{preset}/bin (build first)")
    return exe


def run(cmd, quiet=True):
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=quiet, text=True)
    if r.returncode != 0 and quiet:
        sys.stdout.write(r.stdout + r.stderr)
    return r.returncode


def validate(exe, case, scheme, steps, tol, front_cells, refresh=False, extra_ref=(), extra_cpp=()):
    tag = f"{case}_{scheme}_{steps}"
    ref = ROOT / "out" / f"ref_{tag}"
    cpp = ROOT / "out" / f"cpp_{tag}"
    if refresh or not (ref / "meta.json").exists():
        if run([PY, "tools/dump_reference.py", "--case", case, "--scheme", scheme, "--steps", steps, "--out", ref,
                *extra_ref]):
            return False
    if run([exe, "--headless", "--no-validation", "--case", case, "--scheme", scheme, "--steps", steps, "--dump", cpp,
            *extra_cpp]):
        return False
    print(f"--- {tag}")
    return run([PY, "tools/compare.py", ref, cpp, "--tol", tol, "--front-cells", front_cells,
                "--front-frac", 0 if front_cells == 0 else 5e-4], quiet=False) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", default="dam_break")
    ap.add_argument("scheme", nargs="?", default="ssprk2")
    ap.add_argument("steps", nargs="?", type=int, default=200)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--front-cells", type=int, default=64)
    ap.add_argument("--refresh", action="store_true", help="recompute the Python references")
    ap.add_argument("--preset", default="debug")
    args = ap.parse_args()
    exe = find_exe(args.preset)

    if args.all:
        results = {}
        results["dam_break euler 50 (strict)"] = validate(exe, "dam_break", "euler", 50, 1e-6, 0, args.refresh)
        for s in SCHEMES:
            results[f"dam_break {s} 200"] = validate(exe, "dam_break", s, 200, args.tol, args.front_cells, args.refresh)
        # real data: inflow / level boundaries + Manning (meuse), outlet + 0.48 M cells (vesdre)
        for c in ("meuse", "vesdre"):
            results[f"{c} ssprk2 100"] = validate(exe, c, "ssprk2", 100, args.tol, args.front_cells, args.refresh)
        # exact inflow on a dry domain: 5 m3/s imposed -> 5 m3/s measured
        out = ROOT / "out" / "cpp_inflow_test"
        ok = run([exe, "--headless", "--no-validation", "--case", "inflow_test", "--steps", 3000, "--dump", out]) == 0
        ok = ok and run([PY, "tools/volume_check.py", out, 5], quiet=False) == 0
        results["inflow_test dry-start 5 m3/s"] = ok
        print("\n=== summary")
        for k, v in results.items():
            print(f"{'PASS' if v else 'FAIL'}  {k}")
        return 0 if all(results.values()) else 1

    ok = validate(exe, args.case, args.scheme, args.steps, args.tol, args.front_cells, args.refresh)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
