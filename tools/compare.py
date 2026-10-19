"""Compare two state dumps (Q.npy, zb.npy, meta.json), e.g. Python reference vs C++/Vulkan.

    python tools/compare.py out/ref_dam_ssprk2_200 out/cpp_dam_ssprk2_200 --tol 1e-5

Errors are relative to the largest magnitude of each reference field (h, zb), and to the largest momentum
magnitude max(|hu|, |hv|) for both momentum components: a component that is ~0 by symmetry (hv in the dam break)
is pure round-off and must not be normalised by itself. Exit code 1 when any exceeds --tol
(or when --exact is given and the arrays are not bit-identical).

--front-cells N tolerates up to N cells above --tol for h/hu/hv (discontinuities amplify round-off at shocks
and wet/dry fronts), as long as the relative L1 error stays below --tol.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(d):
    d = Path(d)
    return np.load(d / "Q.npy"), np.load(d / "zb.npy"), json.loads((d / "meta.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tol", type=float, default=1e-5, help="max relative error per component")
    ap.add_argument("--exact", action="store_true", help="require bit-identical arrays")
    ap.add_argument("--front-cells", type=int, default=0, help="cells allowed above --tol (fronts), L1 still checked")
    ap.add_argument("--front-frac", type=float, default=0.0,
                    help="same as --front-cells as a fraction of the cell count (the larger applies)")
    ap.add_argument("--still-atol", type=float, default=0.0,
                    help="cells where the test is exactly at rest (hu = hv = 0) and the reference moves by less than "
                         "this (m2/s) are not compared in momentum: the Python reference has spurious currents on "
                         "still water over real beds (2.3e-6 m2/s after 1 step, 6e-5 after 100 on meuse), the C++ "
                         "none (exact lake at rest)")
    args = ap.parse_args()

    q_ref, zb_ref, m_ref = load(args.ref)
    q_tst, zb_tst, m_tst = load(args.test)
    print(f"ref : {m_ref.get('source')} {m_ref.get('case')} {m_ref.get('scheme')} steps={m_ref.get('steps')} "
          f"t={m_ref['t']:.7f} dt={m_ref['dt']:.6e}")
    print(f"test: {m_tst.get('source')} {m_tst.get('case')} {m_tst.get('scheme')} steps={m_tst.get('steps')} "
          f"t={m_tst['t']:.7f} dt={m_tst['dt']:.6e}")

    if q_ref.shape != q_tst.shape or zb_ref.shape != zb_tst.shape:
        print(f"FAIL shape mismatch: Q {q_ref.shape} vs {q_tst.shape}, zb {zb_ref.shape} vs {zb_tst.shape}")
        return 1

    ok = True
    if args.exact:
        same = np.array_equal(q_ref.view(np.uint32), q_tst.view(np.uint32)) and \
               np.array_equal(zb_ref.view(np.uint32), zb_tst.view(np.uint32))
        print("bit-identical" if same else "FAIL not bit-identical")
        ok = same

    mom_scale = max(float(np.abs(q_ref[..., 1:3]).max()), 1e-30)
    mom_l1 = max(float(np.abs(q_ref[..., 1:3]).sum()) / 2, 1e-30)
    rows = [("zb", zb_ref, zb_tst, None, None), ("h", q_ref[..., 0], q_tst[..., 0], None, None),
            ("hu", q_ref[..., 1], q_tst[..., 1], mom_scale, mom_l1), ("hv", q_ref[..., 2], q_tst[..., 2], mom_scale, mom_l1)]
    print(f"{'field':>5} {'scale':>11} {'max|diff|':>11} {'rel max':>10} {'rel L1':>10} {'cells>tol':>10}")
    for name, a, b, scale, l1 in rows:
        scale = scale or max(float(np.abs(a).max()), 1e-30)
        l1 = l1 or max(float(np.abs(a).sum()), 1e-30)
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if name in ("hu", "hv") and args.still_atol > 0:
            # at rest in the test: |q| <= 1e-8 m2/s (exactly 0 for still water; 1e-9 noise next to moving water)
            still = (np.abs(q_tst[..., 1:3]).max(axis=-1) <= 1e-8) & (np.abs(q_ref[..., 1:3]).max(axis=-1) <= args.still_atol)
            diff = np.where(still, 0.0, diff)
        rel_max = float(diff.max()) / scale
        rel_l1 = float(diff.sum()) / l1
        n_bad = int((diff > args.tol * scale).sum())
        allowed = 0 if name == "zb" else max(args.front_cells, int(args.front_frac * a.size))
        good = (n_bad <= allowed) and rel_l1 <= args.tol
        flag = "" if good else "  <-- FAIL"
        print(f"{name:>5} {scale:11.4e} {float(diff.max()):11.4e} {rel_max:10.2e} {rel_l1:10.2e} {n_bad:10d}{flag}")
        if not np.isfinite(b).all():
            print(f"FAIL {name}: non-finite values in test")
            ok = False
        ok &= good

    for key in ("t", "dt"):
        rel = abs(m_ref[key] - m_tst[key]) / max(abs(m_ref[key]), 1e-30)
        print(f"{key:>5} rel diff {rel:.2e}")
        ok &= rel <= args.tol

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
