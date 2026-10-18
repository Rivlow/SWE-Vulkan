"""Run the original Python/Taichi solver (../SWE) for N steps and dump its state.

    python tools/dump_reference.py --case dam_break --scheme ssprk2 --steps 200 --out out/ref_dam_ssprk2_200

Writes Q.npy (nx, ny, 3), zb.npy (nx, ny) and meta.json, the same layout as `swe --headless --dump`.
The Python project is not modified: main.py is executed with its `case` / `time_scheme` lines replaced,
inside a stand-in `__main__` module (the solver modules read their globals with `from __main__ import ...`).

--shu-osher: run a copy of the project (out/_swe_shu_osher) whose Runge-Kutta loop is rewritten in Shu-Osher form,
Q <- Q^n + b (Q + dt L(Q) - Q^n), with the operation order of euler_stage in shaders/solver/solver.slang. The C++ solver
uses that form (exact positivity with the drain limiter); the Butcher form of the original differs by round-off
only, but round-off is amplified at shocks (and through dt = f(s_max)), so a 200-step comparison needs this.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path

DEFAULT_SWE = Path(__file__).resolve().parents[2] / "SWE"
ROOT = Path(__file__).resolve().parents[1]

BUTCHER_LOOP = """    for s in ti.static(range(tableau.stages)):
        if ti.static(tableau.is_used(s)):
            # Q_s = Q^n + dt * sum_j A[s][j] K_j   (Q_0 = Q^n : Q contient déjà Q^n)
            if ti.static(s > 0):
                combine(Q, tableau.A[s])
                fix_dry(Q)
            residual(Q, K[s])

    # Q^{n+1} = Q^n + dt * sum_s b[s] K_s
    combine(Q, tableau.b)
    fix_dry(Q)
"""
SHU_OSHER_LOOP = """    # Shu-Osher form (dump_reference.py --shu-osher): same operation order as euler_stage in solver.slang
    for s in ti.static(range(len(SHU_OSHER[time_scheme]))):
        residual(Q, K[0])
        a_s, b_s = ti.static(SHU_OSHER[time_scheme][s])
        for i, j in Q:
            qe = Q[i, j] + dt[None] * K[0][i, j]
            if ti.static(a_s != 0.0):
                Q[i, j] = Q_old[i, j] + b_s * (qe - Q_old[i, j])
            else:
                Q[i, j] = qe
        fix_dry(Q)
"""
SHU_OSHER_TABLE = """SHU_OSHER = {"euler": [(0.0, 1.0)], "ssprk2": [(0.0, 1.0), (0.5, 0.5)],
             "ssprk3": [(0.0, 1.0), (0.75, 0.25), (1.0 / 3, 2.0 / 3)]}
"""


def shu_osher_copy(swe):
    """Copy of the Python project with solver/time_integration.py in Shu-Osher form (refreshed every run)."""
    dst = ROOT / "out" / "_swe_shu_osher"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copy2(swe / "main.py", dst / "main.py")
    for d in ("solver", "cases", "ui", "data"):
        if (swe / d).exists():
            shutil.copytree(swe / d, dst / d, ignore=shutil.ignore_patterns("__pycache__"))
    ti_path = dst / "solver" / "time_integration.py"
    src = ti_path.read_text(encoding="utf-8")
    if BUTCHER_LOOP not in src or "HAS_SINK = sink is not None\n" not in src:
        raise SystemExit("--shu-osher: solver/time_integration.py has changed, update the patch in dump_reference.py")
    src = src.replace(BUTCHER_LOOP, SHU_OSHER_LOOP)
    src = src.replace("HAS_SINK = sink is not None\n", "HAS_SINK = sink is not None\n" + SHU_OSHER_TABLE)
    ti_path.write_text(src, encoding="utf-8")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="dam_break")
    ap.add_argument("--scheme", default="ssprk2")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="vulkan", help="taichi arch: vulkan, cuda, cpu")
    ap.add_argument("--swe", default=str(DEFAULT_SWE), help="path to the Python SWE project")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR",
                    help="override a top-level assignment of main.py (e.g. --set eta_0=0.5)")
    ap.add_argument("--still-water", type=float, default=None, metavar="ETA",
                    help="after init, replace the state by still water at level ETA (lake at rest test)")
    ap.add_argument("--shu-osher", action="store_true", help="Runge-Kutta loop in Shu-Osher form (see above)")
    args = ap.parse_args()

    swe = Path(args.swe).resolve()
    if args.shu_osher:
        if args.scheme not in ("euler", "ssprk2", "ssprk3"):
            raise SystemExit("--shu-osher: only euler, ssprk2, ssprk3 have a Shu-Osher form")
        swe = shu_osher_copy(swe)
    out = Path(args.out).resolve()
    src = (swe / "main.py").read_text(encoding="utf-8")

    def replace(name, expr, text):
        pattern = rf"^(\s*){name}\s*=.*$"
        new, n = re.subn(pattern, lambda m: f"{m.group(1)}{name} = {expr}", text, flags=re.M)
        if n == 0:
            raise SystemExit(f"main.py: no assignment to '{name}' found")
        return new

    src = replace("case", repr(args.case), src)
    src = replace("time_scheme", repr(args.scheme), src)
    src = re.sub(r"ti\.init\(arch=ti\.\w+\)", f"ti.init(arch=ti.{args.arch}, random_seed=0)", src)
    for item in args.set:
        name, expr = item.split("=", 1)
        src = replace(name.strip(), expr.strip(), src)
    # never open the window
    src = src.replace('if __name__ == "__main__":', "if False:")

    os.chdir(swe)  # data/ caches are relative
    sys.path.insert(0, str(swe))
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(swe / "main.py")
    sys.modules["__main__"] = fake_main
    exec(compile(src, str(swe / "main.py"), "exec"), fake_main.__dict__)

    ns = fake_main.__dict__
    import taichi as ti
    import numpy as np
    from solver.time_integration import step, read_time, reset_time  # same module objects main.py imported

    Q, zb, init = ns["Q"], ns["zb"], ns["init"]
    init(Q)
    if args.still_water is not None:
        z = zb.to_numpy()
        q = np.zeros((*z.shape, 3), np.float32)
        q[..., 0] = np.maximum(np.float32(args.still_water) - z, np.float32(0))
        Q.from_numpy(q)
    reset_time()
    ti.sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step(Q)
    ti.sync()
    elapsed = time.perf_counter() - t0
    t, dt = read_time()

    from solver import time_integration
    s_max = float(time_integration.s_max[None])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Q.npy", Q.to_numpy().astype(np.float32))
    np.save(out / "zb.npy", zb.to_numpy().astype(np.float32))
    meta = {"source": f"taichi-{args.arch}" + ("-shu-osher" if args.shu_osher else ""), "case": args.case,
            "scheme": args.scheme, "steps": args.steps, "nx": int(ns["nx"]), "ny": int(ns["ny"]), "t": t, "dt": dt,
            "s_max": s_max}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[ref] {args.case} {args.scheme} {args.steps} steps in {elapsed:.3f} s: t = {t:.6f} s, dt = {dt:.4e} s, "
          f"s_max = {s_max:.4f} -> {out}")


if __name__ == "__main__":
    main()
