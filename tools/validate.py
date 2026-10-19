"""Validate the C++/Vulkan solver against the Python/Taichi reference.

    python tools/validate.py dam_break ssprk2 200            # one run
    python tools/validate.py --all                            # every scheme on dam_break + lake at rest
    python tools/validate.py --all --wolfgpu <repo>           # + groundwater against wolfgpu (its Python env)

For each run: tools/dump_reference.py (cached in out/ref_*), swe --headless --dump, tools/compare.py.
Tolerances: relative L1 <= 1e-5 and at most max(--front-cells, 0.05 % of the cells) above 1e-5 (round-off is
amplified at shocks and wet/dry fronts: Taichi CPU vs Taichi Vulkan shows the same pattern, see README; observed
up to 73 cells in hu on dam_break ssprk2 200).
The bed elevation zb must match exactly (same numpy preprocessing ported to C++).
Multi-stage schemes are compared against the Python in Shu-Osher form (dump_reference.py --shu-osher, cached in
out/refso_*): the C++ integrates in that form, the Butcher form of the original differs by round-off that the
shocks amplify. The C++ runs with the drain limiter off (--drain 0), the Python clamps h < 0.

The C++ runs in --reference mode: the Python's Venkatakrishnan limiter (fed the increment to the neighbour), its
dry / thin-film thresholds, still-water depths by subtraction, depth updates without the rounding carry. One
difference stays by design: the C++ keeps still water exactly at rest, the Python has spurious currents of up to
~2.3e-6 m2/s on real beds (27 000 interior cells of meuse after one step, 6e-5 after 100; C++: none). The cells
where the C++ is exactly at rest and the Python moves by less than 1e-4 m2/s are therefore not compared in momentum
(compare.py --still-atol); every other cell is. Euler 200 steps is no longer compared: forward Euler with the MUSCL
reconstruction is unstable (~5 % growth per step), it amplifies those 1e-6 seeds to 1e-3 within 150 steps; Euler is
checked on 50 steps.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SCHEMES = ["euler", "ssprk2", "ssprk3"]
# options that make the C++ use the Python reference's limiter and thresholds (main.py: H_EPS 1e-4, H_DRY 1e-3)
REFERENCE_MODE = ["--reference"]        # + still-water depths by subtraction, depth updates without carry
STILL_ATOL = 1e-4                       # spurious currents of the Python reference on still water (see above)


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


def same_bits(a, b):
    import numpy as np
    qa, qb = np.load(a), np.load(b)
    same = qa.shape == qb.shape and np.array_equal(qa.view(np.uint32), qb.view(np.uint32))
    if not same:
        print(f"bitwise mismatch: {a} vs {b}")
    return same


def validate(exe, case, scheme, steps, tol, front_cells, refresh=False, extra_ref=(), extra_cpp=()):
    tag = f"{case}_{scheme}_{steps}"
    shu_osher = scheme != "euler"        # Euler is the same in both forms: keep the untouched Python as reference
    ref = ROOT / "out" / f"{'refso' if shu_osher else 'ref'}_{tag}"
    cpp = ROOT / "out" / f"cpp_{tag}"
    if refresh or not (ref / "meta.json").exists():
        if run([PY, "tools/dump_reference.py", "--case", case, "--scheme", scheme, "--steps", steps, "--out", ref,
                *(["--shu-osher"] if shu_osher else []), *extra_ref]):
            return False
    # the Python reference clamps h < 0: compare with the drain limiter off; and with its Venkatakrishnan limiter
    # (fed the increment to the neighbour) and its dry / thin-film thresholds, which the C++ no longer uses by default
    if run([exe, "--headless", "--no-validation", "--case", case, "--scheme", scheme, "--steps", steps, "--drain", 0,
            *REFERENCE_MODE, "--dump", cpp, *extra_cpp]):
        return False
    print(f"--- {tag}")
    return run([PY, "tools/compare.py", ref, cpp, "--tol", tol, "--front-cells", front_cells,
                "--front-frac", 0 if front_cells == 0 else 5e-4, "--still-atol", STILL_ATOL], quiet=False) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", default="dam_break")
    ap.add_argument("scheme", nargs="?", default="ssprk2")
    ap.add_argument("steps", nargs="?", type=int, default=200)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--front-cells", type=int, default=128)
    ap.add_argument("--refresh", action="store_true", help="recompute the Python references")
    ap.add_argument("--preset", default="debug")
    ap.add_argument("--wolfgpu", type=Path, help="wolfgpu repository: also compare the groundwater layer with it "
                                                "(tools/gw_compare_wolfgpu.py, needs its Python environment)")
    args = ap.parse_args()
    exe = find_exe(args.preset)
    wolfgpu, preset = args.wolfgpu, args.preset        # `args` is reused by a loop below

    if args.all:
        results = {}
        # strict: no front-cell allowance; 2e-6 rather than 1e-6 since the Python's spurious currents over the tail of
        # the bump (z ~ 1e-7 m at the dam) are carried by the rarefaction (observed 1.1e-6 in 4 cells)
        results["dam_break euler 50 (strict)"] = validate(exe, "dam_break", "euler", 50, 2e-6, 0, args.refresh)
        for s in SCHEMES[1:]:        # (euler 200: unstable with MUSCL, see the module docstring)
            results[f"dam_break {s} 200"] = validate(exe, "dam_break", s, 200, args.tol, args.front_cells, args.refresh)
        # real data: inflow / level boundaries + Manning (meuse), outlet + 0.48 M cells (vesdre)
        for c in ("meuse", "vesdre"):
            results[f"{c} ssprk2 100"] = validate(exe, c, "ssprk2", 100, args.tol, args.front_cells, args.refresh)
        # exact inflow on a dry domain: 5 m3/s imposed -> 5 m3/s measured
        out = ROOT / "out" / "cpp_inflow_test"
        ok = run([exe, "--headless", "--no-validation", "--case", "inflow_test", "--steps", 3000, "--dump", out]) == 0
        ok = ok and run([PY, "tools/volume_check.py", out, 5], quiet=False) == 0
        results["inflow_test dry-start 5 m3/s"] = ok
        # drain limiter. drain_test (dam break, Euler, CFL 0.9) does go negative without it: check that first,
        # then with the limiter (case default and 1 pass): h >= 0 everywhere, no clamp, volume conserved between walls
        out = ROOT / "out" / "cpp_drain_test_off"
        ok = run([exe, "--headless", "--no-validation", "--case", "drain_test", "--steps", 1000, "--drain", 0, "--dump", out]) == 0
        ok = ok and run([PY, "tools/positivity_check.py", out, "--expect-clamps"], quiet=False) == 0
        results["drain_test limiter off: clamps happen"] = ok
        for c, s, steps, extra in (("drain_test", "euler", 1000, ()), ("drain_test", "euler", 1000, ("--drain", 1)),
                                   ("drain_test", "ssprk3", 1000, ()), ("dam_break", "ssprk2", 200, ()),
                                   ("lake_at_rest", "ssprk2", 200, ())):
            out = ROOT / "out" / f"cpp_drain_{c}_{s}_{steps}{'_' + str(extra[1]) if extra else ''}"
            ok = run([exe, "--headless", "--no-validation", "--case", c, "--scheme", s, "--steps", steps, *extra, "--dump", out]) == 0
            ok = ok and run([PY, "tools/positivity_check.py", out, "--conserve"], quiet=False) == 0
            results[f"{c} {s} {steps} {' '.join(map(str, extra)) or 'default'} drain limiter: h >= 0, mass"] = ok
        out = ROOT / "out" / "cpp_drain_vesdre_ssprk2_100"
        ok = run([exe, "--headless", "--no-validation", "--case", "vesdre", "--steps", 100, "--dump", out]) == 0
        ok = ok and run([PY, "tools/positivity_check.py", out], quiet=False) == 0
        results["vesdre ssprk2 100 drain limiter: h >= 0"] = ok
        # active tiles: same bits as computing every tile (fronts crossing dry tiles, sources, levels, inflow)
        for name, args in (("inflow_test ssprk3", ["--case", "inflow_test", "--scheme", "ssprk3", "--steps", 2000]),
                           ("vesdre", ["--case", "vesdre", "--steps", 600]),
                           ("dam_break + blocks + zones", ["--case", "dam_break", "--scheme", "euler", "--steps", 600,
                                                           "--water", "200,16,215,31,0.5", "--zone", "source,0.2,20,200,24,204",
                                                           "--zone", "level,0.3,230,230,240,240"])):
            tag = name.split()[0]
            outs = [ROOT / "out" / f"cpp_tiles_{tag}_{mode}" for mode in ("active", "all")]
            ok = run([exe, "--headless", "--no-validation", *args, "--dump", outs[0]]) == 0
            ok = ok and run([exe, "--headless", "--no-validation", *args, "--all-tiles", "--dump", outs[1]]) == 0
            ok = ok and same_bits(outs[0] / "Q.npy", outs[1] / "Q.npy")
            results[f"{name}: active tiles == all tiles (bitwise)"] = ok
        # groundwater: positivity and mass balance surface + aquifer; exfiltration wets dry tiles (gw_seepage),
        # so the aquifer state must match the full-grid run too
        seepage = ["--case", "gw_seepage", "--water", "100,100,140,140,0.5", "--steps", 20000]
        outs = [ROOT / "out" / f"cpp_gw_seepage_{mode}" for mode in ("active", "all")]
        ok = run([exe, "--headless", "--no-validation", *seepage, "--dump", outs[0]]) == 0
        ok = ok and run([PY, "tools/groundwater_check.py", outs[0], "--expect-exfiltration"], quiet=False) == 0
        results["gw_seepage: h, h_gw >= 0, surface + aquifer mass, exfiltration"] = ok
        ok = run([exe, "--headless", "--no-validation", *seepage, "--all-tiles", "--dump", outs[1]]) == 0
        ok = ok and same_bits(outs[0] / "Q.npy", outs[1] / "Q.npy") and same_bits(outs[0] / "gw.npy", outs[1] / "gw.npy")
        results["gw_seepage: active tiles == all tiles (bitwise, Q and gw)"] = ok
        out = ROOT / "out" / "cpp_gw_wolfgpu"
        ok = run([exe, "--headless", "--no-validation", "--case", "gw_wolfgpu", "--steps", 3000, "--dump", out]) == 0
        ok = ok and run([PY, "tools/groundwater_check.py", out], quiet=False) == 0
        results["gw_wolfgpu (wolfgpu convention): mass surface + aquifer"] = ok
        # infiltration / rain chronologies on zones (+ aquifer): rain volume of the table (2 zones, interpolated:
        # 50 m2 x 3000 mm/h x 20 s + 50 m2 x 1500 mm/h x 20 s + 50 m2 x 3000 mm/h x 40 s = 2.9167 m3), balance,
        # rain on dry tiles == full grid
        rain = ["--case", "rain_infiltration", "--steps", 20000]
        outs = [ROOT / "out" / f"cpp_rain_{mode}" for mode in ("active", "all")]
        ok = run([exe, "--headless", "--no-validation", *rain, "--dump", outs[0]]) == 0
        ok = ok and run([PY, "tools/groundwater_check.py", outs[0], "--rain", 35 / 12], quiet=False) == 0
        results["rain_infiltration: rain volume, h, h_gw >= 0, mass"] = ok
        ok = run([exe, "--headless", "--no-validation", *rain, "--all-tiles", "--dump", outs[1]]) == 0
        ok = ok and all(same_bits(outs[0] / f, outs[1] / f) for f in ("Q.npy", "gw.npy", "sources.npy"))
        results["rain_infiltration: active tiles == all tiles (bitwise)"] = ok
        if wolfgpu:
            results["groundwater vs wolfgpu (h_gw rel L1 <= 1e-3)"] = run([PY, "tools/gw_compare_wolfgpu.py", "--wolfgpu", wolfgpu,
                                                                     "--preset", preset], quiet=False) == 0
        print("\n=== summary")
        for k, v in results.items():
            print(f"{'PASS' if v else 'FAIL'}  {k}")
        return 0 if all(results.values()) else 1

    ok = validate(exe, args.case, args.scheme, args.steps, args.tol, args.front_cells, args.refresh)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
