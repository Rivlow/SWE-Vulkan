"""Comparisons with analytical solutions (1-D problems on 3-cell-wide channels between walls):
- dam breaks of Ritter (dry bed) and Stoker (wet bed), with a grid-convergence check
  (wolfgpu Analysis_linear_reconstruction.py "ritter", without assertion there);
- SWASHES bumps (Delestre et al. 2013): subcritical, transcritical, transcritical with a hydraulic jump
  (wolfgpu Analysis_linear_reconstruction.py "parabolic_bump_no_jump");
- gradually varied flow M1 / M2 against the integrated backwater equation (wolfgpu Analysis_flux_formulation.py).
Mass is checked on every run."""
import numpy as np
import pytest

import swetest
from swetest import assert_mass

W = 3        # channel width in cells


def _dam(name, L, cell, x0, h_l, h_r):
    nx = int(round(L / cell))
    x = swetest.centres(nx, cell)
    h = np.where(x < x0, h_l, h_r)
    return swetest.pool(name, nx, W, cell, h=np.repeat(h[:, None], W, axis=1)), x


def _dam_error(run, kind, cell, scheme="ssprk2"):
    L, x0, t_end, h_l = 100.0, 40.0, 4.0, 1.0
    h_r = 0.0 if kind == "ritter" else 0.1
    c, x = _dam(f"{kind}_{cell}", L, cell, x0, h_l, h_r)
    r = run(c, time=t_end, scheme=scheme, name=f"{kind}_{cell}")
    assert_mass(r, tol=1e-6, volume0=c.volume0())
    h = r.h[:, 1]
    ref, _ = (swetest.ritter(x, r.t, h_l, x0) if kind == "ritter" else swetest.stoker(x, r.t, h_l, h_r, x0))
    return float(np.abs(h - ref).sum() / np.abs(ref).sum()), h, ref, x, r


def _front(kind, x, h, ref, level=None):
    """Ritter: last cell with h > level (wet front); Stoker: last cell above the mid-height of the shock."""
    if kind == "ritter":
        return x[h > level].max(), x[ref > level].max()
    h_m = float(ref[(ref > 0.1 + 1e-9) & (x > 40.0)].min())        # middle state
    mid = 0.5 * (h_m + 0.1)
    return x[h > mid].max(), x[ref > mid].max()


@pytest.mark.parametrize("kind", ["ritter", "stoker"])
def test_dam_break(run, plot, kind):
    """L1 error (relative) at dx = 0.1 m, convergence when dx is halved, front position.
    Ritter, measured (dx = 0.1, 0.05 m): L1 9.9e-4, 4.8e-4; front h > 1 cm 0.2 m, 0.1 m behind the exact one,
    h > 1 mm 1.3 m, 0.8 m behind (the very tip, h -> 0, is smeared by any scheme). With the former Venkatakrishnan
    feed (phi = 0.75 on smooth data) and thin-film threshold (1 mm) the front stayed 3.5 m behind whatever dx."""
    e_coarse, h_c, ref_c, x_c, _ = _dam_error(run, kind, 0.1)
    e_fine, h, ref, x, r = _dam_error(run, kind, 0.05)
    # metric: relative L1 error sum|h - h_exact| / sum|h_exact| at t = 4 s, and the front position
    plot.lines("profile", [x_c, x, x], [("swe dx = 0.1 m", h_c, "aux"), ("swe dx = 0.05 m", h, "num"), (f"exact ({kind})", ref, "ref")],
               title=f"Dam break {kind}: depth at t = {r.t:.1f} s", ylabel="h (m)", xlim=(20, 80),
               note=f"relative L1: {e_coarse:.2e} (dx 0.1) -> {e_fine:.2e} (dx 0.05), limit 1e-3")
    plot.error("error", x, h - ref, title=f"{kind}: |h - h_exact| (large only at the front / shock)")
    plot.convergence("convergence", [0.1, 0.05], [e_coarse, e_fine], tol=1e-3, title=f"{kind}: L1 error against dx")
    assert e_fine <= 1e-3, f"relative L1 error {e_fine:.3e}"
    assert e_fine < 0.7 * e_coarse, f"no convergence: {e_coarse:.3e} -> {e_fine:.3e}"
    if kind == "ritter":
        for level, tol in ((1e-2, 0.3), (1e-3, 1.2)):
            front, exact = _front(kind, x, h, ref, level)
            front_c, _ = _front(kind, x_c, h_c, ref_c, level)
            assert abs(front - exact) <= tol, f"front h > {level}: {front:.2f} m, exact {exact:.2f} m"
            assert abs(front - exact) <= abs(front_c - exact) + 1e-9, "the front does not get closer when dx is halved"
    else:
        front, exact = _front(kind, x, h, ref)
        assert abs(front - exact) <= 0.2, f"shock at {front:.2f} m, exact {exact:.2f} m"


def _bump_case(name, q, bc_east, eta0, cell=0.1):
    nx = int(round(25.0 / cell))
    zb = swetest.bump_bed(nx, cell, W)
    h0 = np.maximum(eta0 - zb, 0.0)
    bc = dict(swetest.WALLS)
    bc["west"] = ["inflow", q * W * cell]
    bc["east"] = bc_east
    return swetest.Case(name, cell, zb, h0, bc=bc)


def _bump_plot(plot, name, x, zb, h, ref, q_num, q, e, tol, title, vlines=()):
    plot.lines(name, x, [("bed zb", zb, "aux"), ("swe: surface", h + zb, "num"), ("exact: surface", ref + zb, "ref")],
               title=title, ylabel="elevation (m)", vlines=vlines, note=f"relative L1 of h: {e:.2e} (limit {tol:.0e})")
    plot.lines(name + "_discharge", x, [("swe: q = hu", q_num, "num"), ("imposed q", np.full_like(x, q), "ref")],
               title="Unit discharge (must be constant in steady state)", ylabel="q (m2/s)",
               note=f"max |hu - q| = {np.abs(q_num - q).max():.2e} m2/s")


def test_bump_subcritical(run, plot):
    q, h_out = 4.42, 2.0
    c = _bump_case("bump_sub", q, ["depth", h_out], 2.0)
    r = run(c, time=150.0)
    swetest.assert_positive(r)
    x = swetest.centres(c.nx, c.cell)
    ref = swetest.bump_subcritical(x, c.zb[:, 1], q, h_out)
    h = r.h[:, 1]
    _bump_plot(plot, "bump", x, c.zb[:, 1], h, ref, r.hu[:, 1], q, swetest.rel_l1(ref, h), 2e-3, "SWASHES 3.1.1: subcritical flow over the bump")
    assert swetest.rel_l1(ref, h) <= 2e-3, f"relative L1 {swetest.rel_l1(ref, h):.3e}"
    assert np.abs(r.hu[:, 1] - q).max() <= 5e-3 * q


def test_bump_transcritical(run, plot):
    """SWASHES: h = 0.66 m imposed downstream "if the flow is subcritical" - a depth condition, which the HLLC flux
    ignores once the outflow is supercritical (upwind). (A zero-gradient outflow instead lets the reach fill up to
    a drowned subcritical state: no control.)"""
    q = 1.53
    c = _bump_case("bump_trans", q, ["depth", 0.66], 0.66)
    r = run(c, time=150.0)
    swetest.assert_positive(r)
    x = swetest.centres(c.nx, c.cell)
    ref = swetest.bump_transcritical(x, c.zb[:, 1], q)
    h = r.h[:, 1]
    _bump_plot(plot, "bump", x, c.zb[:, 1], h, ref, r.hu[:, 1], q, swetest.rel_l1(ref, h), 5e-3, "SWASHES 3.1.2: transcritical flow without shock")
    assert swetest.rel_l1(ref, h) <= 5e-3, f"relative L1 {swetest.rel_l1(ref, h):.3e}"
    assert np.abs(r.hu[5:-5, 1] - q).max() <= 1e-2 * q


def test_bump_hydraulic_jump(run, plot):
    q, h_out = 0.18, 0.33
    c = _bump_case("bump_jump", q, ["depth", h_out], 0.33)
    r = run(c, time=300.0)
    swetest.assert_positive(r)
    x = swetest.centres(c.nx, c.cell)
    ref, x_jump = swetest.bump_shock(x, c.zb[:, 1], q, h_out)
    h = r.h[:, 1]
    _bump_plot(plot, "bump", x, c.zb[:, 1], h, ref, r.hu[:, 1], q, swetest.rel_l1(ref, h), 1e-2,
               "SWASHES 3.1.3: transcritical flow with a hydraulic jump", vlines=[(x_jump, "exact jump")])
    assert swetest.rel_l1(ref, h) <= 1e-2, f"relative L1 {swetest.rel_l1(ref, h):.3e}"
    # numerical jump: largest increase of h downstream of the crest
    i_c = int(np.argmax(c.zb[:, 1]))
    i_j = i_c + int(np.argmax(np.diff(h[i_c:])))
    assert abs(x[i_j] - x_jump) <= 0.3, f"jump at {x[i_j]:.2f} m, exact {x_jump:.2f} m"


@pytest.mark.parametrize("h_down,kind", [(0.7, "M2"), (1.5, "M1")])
def test_backwater_curve(run, plot, h_down, kind):
    """Mild slope (h_n = 0.969 m > h_c = 0.467 m), downstream depth below (M2, drawdown) or above (M1, backwater)
    the normal depth: steady profile against dh/dx = (S0 - Sf) / (1 - Fr^2)."""
    q, n, s, cell, nx = 1.0, 0.03, 1e-3, 5.0, 300
    hn = swetest.normal_depth(q, n, s)
    c = swetest.channel(f"backwater_{kind}", nx, W, cell, slope=s, h=hn, q=q, manning=n,
                        bc={"west": ["inflow", q * W * cell], "east": ["depth", h_down]})
    r = run(c, time=20000.0)
    assert_mass(r, tol=1e-5)
    x = swetest.centres(nx, cell)
    h = r.h[:, 1]
    # the weak depth condition sets the ghost state (with the cell's velocity), not the cell: the last cell sits a
    # few % off h_down (observed +3.4 % on the M2 drawdown, Fr = 0.55 there)
    assert abs(h[-1] - h_down) <= 5e-2 * h_down, f"downstream depth {h[-1]:.4f} m for {h_down} imposed"
    # the profile itself: the backwater equation integrated upstream from the computed downstream depth
    ref = swetest.backwater_m2(x, q, n, s, float(h[-1]))
    err = np.abs(h - ref)
    plot.lines("profile", x, [("bed", c.zb[:, 1], "aux"), ("swe: surface", h + c.zb[:, 1], "num"), ("backwater equation: surface", ref + c.zb[:, 1], "ref"),
                              ("normal depth hn above the bed", c.zb[:, 1] + hn, "aux")],
               title=f"Gradually varied flow {kind} (q = {q} m2/s, n = {n}, S0 = {s})", ylabel="elevation (m)",
               note=f"max |h - h_ref| = {err.max():.2e} m (limit {5e-3 * h_down:.1e}); h_down imposed {h_down}, computed {h[-1]:.3f}")
    plot.error("error", x, err, tol=5e-3 * h_down, title=f"{kind}: |h - h_backwater|", xlabel="x (m)")
    assert err.max() <= 5e-3 * h_down, f"max |h - h_ref| = {err.max():.4f} m"
    assert swetest.rel_l1(ref, h) <= 2e-3
