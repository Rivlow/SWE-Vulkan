"""Compare the groundwater layer of SWE-Vulkan with wolfgpu's on the same problem.

    python tools/gw_compare_wolfgpu.py [--wolfgpu C:/Users/lucas/Dev/wolfgpu/Repository] [--time 600] [--preset release]

Run with the Python environment of wolfgpu (e.g. the conda env wolf_env): wolfgpu needs OpenGL (an invisible
GLFW window).

Problem (wolfgpu's convention, "datum": "wolfgpu" on the SWE-Vulkan side): flat bed at 1 m, 64 x 64 cells of 1 m
between walls, 0.2 m of still water everywhere, and a Gaussian mound of the aquifer under it. Both layers move:
the surface water infiltrates at the constant rate kv (zb + h > h_gw everywhere), the mound spreads laterally
(Boussinesq, kh). The surface stays nearly flat, so the comparison isolates the groundwater model.

Prints the volumes and the differences of h_gw and h at the final time (relative L1, max), writes both fields to
out/gw_compare/. Exit code 0 when h_gw agrees within --tol (relative L1).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

N = 64            # interior cells per side
DX = 1.0
BED = 1.0
H0 = 0.2
KV = 1e-4
KH = 5e-2


def fields():
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r2 = ((i - N / 2 + 0.5) ** 2 + (j - N / 2 + 0.5) ** 2) / (N / 8) ** 2
    h_gw = (0.2 + 0.6 * np.exp(-r2)).astype(np.float32)
    return h_gw


def run_wolfgpu(repo: Path, t_end: float, out: Path):
    sys.path.insert(0, str(repo))
    import glfw  # noqa: F401  (initialised by the harness)
    from wolfgpu.simple_simulation import SimpleSimulation
    from wolfgpu.simplesimu.durations import SimulationDuration
    from wolfgpu.results_store import ResultsStore
    from tools_for_analysis.Manager.simulation_runner_base import SimulationTestBase

    nx = ny = N + 2        # a ring of walls (NAP = 0, bed 99999) around the interior
    nap = np.zeros((nx, ny), dtype=np.uint8)
    nap[1:-1, 1:-1] = 1
    bathy = np.full((nx, ny), 99999.0, dtype=np.float32)
    bathy[1:-1, 1:-1] = BED
    h = np.zeros((nx, ny), dtype=np.float32)
    h[1:-1, 1:-1] = H0
    h_gw = np.zeros((nx, ny), dtype=np.float32)
    h_gw[1:-1, 1:-1] = fields()

    sim = SimpleSimulation(nx=nx, ny=ny)
    sim.h = h
    sim.qx = np.zeros((nx, ny), dtype=np.float32)
    sim.qy = np.zeros((nx, ny), dtype=np.float32)
    sim.manning = np.zeros((nx, ny), dtype=np.float32)
    sim.nap = nap
    sim.bathymetry = bathy
    sim.param_dx = DX
    sim.param_dy = DX
    sim.param_courant = 0.4
    sim.param_duration = SimulationDuration.from_seconds(t_end)
    sim.param_report_period = SimulationDuration.from_seconds(t_end)
    zones = np.zeros((nx, ny), dtype=np.int32)
    zones[1:-1, 1:-1] = 1
    sim.infiltration_zones = zones
    sim.add_infiltration(0.0, [0.0])
    kv = np.zeros((nx, ny), dtype=np.float32)
    kv[1:-1, 1:-1] = KV
    sim.darcy_K = kv
    kh = np.zeros((nx, ny), dtype=np.float32)
    kh[1:-1, 1:-1] = KH
    sim.Kh_darcy = kh
    sim.h0_gw = h_gw
    sim.check_errors()

    class Harness(SimulationTestBase):
        def runTest(self):
            pass

    runner = Harness()._run_simple_sim(sim, window=False, early_out_delta_max=None, record_path=out / "wolfgpu_run")
    rs = ResultsStore(runner.results_path(), "r")
    n = rs.nb_results
    t = float(np.array(rs.get_named_series("t"))[-1])
    h_end = np.asarray(rs.get_named_result("h", n, untile=True))[1:-1, 1:-1]
    gw_end = np.asarray(rs.get_named_result("h_gw", n, untile=True))[1:-1, 1:-1]
    return t, h_end.astype(np.float64), gw_end.astype(np.float64)


def run_swe(preset: str, t_end: float, out: Path):
    exe = ROOT / "build" / preset / "bin" / "swe.exe"
    assets = out / "assets"
    (assets / "cases").mkdir(parents=True, exist_ok=True)
    (assets / "data").mkdir(parents=True, exist_ok=True)
    full = np.full((N, N), 1.0, dtype=np.float32)
    np.savez(assets / "data" / "gw_compare.npz", zb=full * BED, h0=full * H0, gw_kv=full * KV, gw_kh=full * KH,
             gw_h0=fields())
    case = {"label": "wolfgpu comparison", "type": "imported", "data": "data/gw_compare.npz", "cell": DX,
            "manning_n": 0.0, "scheme": "ssprk2",
            "groundwater": {"datum": "wolfgpu", "kv": "gw_kv", "kh": "gw_kh", "h0": "gw_h0"}}
    (assets / "cases" / "gw_compare.json").write_text(json.dumps(case, indent=2))
    dump = out / "swe_run"
    r = subprocess.run([str(exe), "--headless", "--no-validation", "--assets", str(assets), "--case", "gw_compare",
                        "--time", str(t_end), "--dump", str(dump)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout + r.stderr)
        raise SystemExit("swe failed")
    meta = json.loads((dump / "meta.json").read_text())
    q = np.load(dump / "Q.npy")
    gw = np.load(dump / "gw.npy")
    return float(meta["t"]), q[..., 0].astype(np.float64), gw[..., 0].astype(np.float64)


def report(name, a, b):
    rel = np.abs(a - b).sum() / max(np.abs(a).sum(), 1e-30)
    print(f"  {name:5s} rel L1 {rel:.3e}   max |diff| {np.abs(a - b).max():.3e} m   (max {np.abs(a).max():.3f} m)")
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wolfgpu", type=Path, default=Path("C:/Users/lucas/Dev/wolfgpu/Repository"))
    ap.add_argument("--time", type=float, default=600.0)
    ap.add_argument("--preset", default="release")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative L1 tolerance on h_gw (observed: 3e-5)")
    args = ap.parse_args()
    out = ROOT / "out" / "gw_compare"
    out.mkdir(parents=True, exist_ok=True)

    t_w, h_w, gw_w = run_wolfgpu(args.wolfgpu, args.time, out)
    t_s, h_s, gw_s = run_swe(args.preset, args.time, out)
    np.savez(out / "fields.npz", h_wolfgpu=h_w, gw_wolfgpu=gw_w, h_swe=h_s, gw_swe=gw_s)
    gw0 = fields().astype(np.float64)
    area = DX * DX
    print(f"t: wolfgpu {t_w:.3f} s, SWE-Vulkan {t_s:.3f} s")
    print(f"aquifer volume: initial {gw0.sum() * area:.4f} m3, wolfgpu {gw_w.sum() * area:.4f}, SWE-Vulkan {gw_s.sum() * area:.4f}")
    print(f"surface volume: initial {H0 * N * N * area:.4f} m3, wolfgpu {h_w.sum() * area:.4f}, SWE-Vulkan {h_s.sum() * area:.4f}")
    print(f"total: wolfgpu {(gw_w + h_w).sum() * area:.4f}, SWE-Vulkan {(gw_s + h_s).sum() * area:.4f} "
          f"(initial {(gw0.sum() + H0 * N * N) * area:.4f})")
    print("differences at the final time:")
    rel_gw = report("h_gw", gw_s, gw_w)
    report("h", h_s, h_w)
    # what the aquifer change itself looks like, for scale
    report("dh_gw", gw_s - gw0, gw_w - gw0)
    ok = rel_gw <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
