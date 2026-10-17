"""Water volume of a dump: python tools/volume_check.py out/inflow_test [expected_discharge_m3s]"""
import json
import sys

import numpy as np

d = sys.argv[1]
q = np.load(f"{d}/Q.npy")
m = json.load(open(f"{d}/meta.json"))
cell = 10.0 / m["nx"] if "cell" not in m else m["cell"]
V = float(q[..., 0].sum()) * cell * cell
h = q[..., 0]
print(f"t = {m['t']:.3f} s, volume = {V:.3f} m3, mean discharge = {V / max(m['t'], 1e-9):.3f} m3/s, "
      f"wet cells = {(h > 1e-3).sum()}, h max = {h.max():.3f} m")
if len(sys.argv) > 2:
    Q = float(sys.argv[2])
    ok = abs(V / m["t"] - Q) / Q < 0.02
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
