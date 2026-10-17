"""Run the original Python/Taichi solver (../SWE) for N steps and dump its state.

    python tools/dump_reference.py --case dam_break --scheme ssprk2 --steps 200 --out out/ref_dam_ssprk2_200

Writes Q.npy (nx, ny, 3), zb.npy (nx, ny) and meta.json, the same layout as `swe --headless --dump`.
The Python project is not modified: main.py is executed with its `case` / `time_scheme` lines replaced,
inside a stand-in `__main__` module (the solver modules read their globals with `from __main__ import ...`).
"""
import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path

DEFAULT_SWE = Path(__file__).resolve().parents[2] / "SWE"


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
    args = ap.parse_args()

    swe = Path(args.swe).resolve()
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
    meta = {"source": f"taichi-{args.arch}", "case": args.case, "scheme": args.scheme, "steps": args.steps,
            "nx": int(ns["nx"]), "ny": int(ns["ny"]), "t": t, "dt": dt, "s_max": s_max}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[ref] {args.case} {args.scheme} {args.steps} steps in {elapsed:.3f} s: t = {t:.6f} s, dt = {dt:.4e} s, "
          f"s_max = {s_max:.4f} -> {out}")


if __name__ == "__main__":
    main()
