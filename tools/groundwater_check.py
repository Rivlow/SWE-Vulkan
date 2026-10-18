"""Check a run with vertical terms dumped by swe --headless --dump: positivity and mass balance.

    python tools/groundwater_check.py out/cpp_gw_seepage [--tol 1e-4] [--expect-exfiltration] [--rain 2.9167]

Groundwater (gw.npy = h_gw, exchange rate, cumulated infiltration, cumulated exfiltration) and / or infiltration and
rain chronologies (sources.npy = net depth added to the surface, rain depth):
- h >= 0 and h_gw >= 0 everywhere;
- between walls, surface + aquifer - volume added by the chronologies is conserved (relative change <= --tol,
  float32 accumulation over the run);
- the surface change matches the cumulated terms: V_sw - V_sw0 == exfiltrated - infiltrated + added;
- --rain: the rain volume of the chronology (m3), within --tol relative.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--expect-exfiltration", action="store_true", help="the run must bring water back to the surface")
    ap.add_argument("--rain", type=float, help="expected rain volume (m3)")
    ap.add_argument("--rain-tol", type=float, default=1e-3, help="relative tolerance on the rain volume (time steps)")
    args = ap.parse_args()

    meta = json.loads((args.dump / "meta.json").read_text())
    q = np.load(args.dump / "Q.npy")
    has_gw = (args.dump / "gw.npy").exists()
    ok = True

    h_min = float(q[..., 0].min())
    hgw_min = float(np.load(args.dump / "gw.npy")[..., 0].min()) if has_gw else 0.0
    print(f"h min = {h_min:.3e} m" + (f", h_gw min = {hgw_min:.3e} m" if has_gw else ""))
    if h_min < 0.0 or hgw_min < 0.0:
        print("FAIL: negative depth")
        ok = False

    v0, v = meta["volume0"], meta["volume"]
    g0, g = meta.get("gw_volume0", 0.0), meta.get("gw_volume", 0.0)
    infil, exfil = meta.get("gw_infiltrated", 0.0), meta.get("gw_exfiltrated", 0.0)
    added, rain = meta.get("sources_net", 0.0), meta.get("rain_volume", 0.0)
    print(f"surface {v0:.6f} -> {v:.6f} m3, aquifer {g0:.6f} -> {g:.6f} m3, infiltrated {infil:.6f}, "
          f"exfiltrated {exfil:.6f}, added by the chronologies {added:+.6f} (rain {rain:.6f})")
    total0, total = v0 + g0, v + g - added
    rel = (total - total0) / max(total0, v + g)
    print(f"surface + aquifer - added: {rel:+.3e} relative")
    if abs(rel) > args.tol:
        print(f"FAIL: total volume not conserved (tolerance {args.tol:g})")
        ok = False
    residual = (v - v0) - (exfil - infil) - added
    print(f"surface change - (exfiltrated - infiltrated) - added = {residual:+.3e} m3")
    if abs(residual) > args.tol * max(v0, v, abs(added), 1e-12):
        print("FAIL: surface change does not match the cumulated terms")
        ok = False
    if has_gw and infil <= 0.0:
        print("FAIL: nothing infiltrated")
        ok = False
    if args.expect_exfiltration and exfil <= 0.0:
        print("FAIL: no exfiltration")
        ok = False
    if args.rain is not None and abs(rain - args.rain) > args.rain_tol * args.rain:
        print(f"FAIL: rain volume {rain:.6f} m3, expected {args.rain:.6f}")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
