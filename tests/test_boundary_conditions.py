"""Boundary conditions (wolfgpu test_boundary_conditions.py: hmod / h symmetry, imposed levels pulling the water
down or holding it; weak conditions on faces). Every test closes the mass balance with the volumes the solver
measures through each boundary (exact), and checks what flowed through which boundary."""
import numpy as np
import pytest

import swepost
import swetest
from swetest import assert_mass


def test_free_outflow_drains(run):
    """Tilted pool open on the east (free outflow): the volume decreases monotonically, everything leaves through
    the east side, and the post-processed hydrogram agrees (for a free side the boundary mass flux is the boundary
    cell's own hu, which swepost reads: only the time sampling of Q(t) differs)."""
    nx, ny = 80, 3
    c = swetest.channel("drain_free", nx, ny, 1.0, slope=2e-3, h=0.5, bc={"east": ["free"]})
    r = run(c, time=300.0, record_every=0.25)
    assert_mass(r, tol=1e-6)
    east = r.meta["boundary_volume"][1]
    # a zero-gradient outflow is no control section: in subcritical flow the outflow stops once the surface is
    # flat (22 m3 of 120 leave, then nothing), it does not empty the pool
    assert east < -0.1 * c.volume0()
    assert r.meta["boundary_total"] == pytest.approx(east)
    b = swepost.hydrograph_balance(r.rec, outflows=[("x", nx, 0, ny - 1)])
    r.plot.lines("volume", b["t"], [("V(t) from the recording", b["V"], "num"), ("final volume expected: V0 + east side (exact)", np.full_like(b["t"], c.volume0() + east), "ref")],
                 xlabel="t (s)", ylabel="volume (m3)", title="Free outflow: the pool drains, then stops when the surface is flat",
                 note=f"east side removed {-east:.1f} m3 of {c.volume0():.1f}; hydrograph residual max {np.abs(b['residual']).max():.2e} m3")
    r.plot.lines("hydrograph", b["t"], [("Q out (post-processed)", b["Qout"], "num")], xlabel="t (s)", ylabel="Q (m3/s)", title="Outflow hydrograph")
    # monotone, up to float32 round-off of the depths once the outflow has stopped (1 ulp of h ~ 1e-7 V)
    assert np.all(np.diff(b["V"]) <= 1e-7 * b["V"][0])
    assert np.abs(b["residual"]).max() <= 5e-3 * abs(east)


def test_level_pull_down_symmetry(run):
    """wolfgpu test_h_symmetry_*: a lower level imposed on one side pulls the water down; the same problem along
    x and along y gives the transposed field, and the water leaves through that side only."""
    def case(name, axis):
        shape = (60, 5) if axis == "x" else (5, 60)
        bc = dict(swetest.WALLS)
        bc["east" if axis == "x" else "north"] = ["level", 0.9]
        return swetest.Case(name, 1.0, np.zeros(shape), np.full(shape, 1.0), bc=bc)

    a = run(case("pull_x", "x"), time=60.0, record_every=5.0, name="x")
    b = run(case("pull_y", "y"), time=60.0, record_every=5.0, name="y")
    for r in (a, b):
        assert_mass(r, tol=1e-6)
    a.plot.lines("symmetry", [swetest.centres(60, 1.0)] * 2, [("along x: h(x)", a.h[:, 2], "num"), ("along y: h(y)", b.h[2, :], "ref")],
                 title="Level 0.9 m imposed on one side: same field along x and along y", ylabel="h (m)",
                 note=f"max |h_x - h_y^T| = {np.abs(a.h - b.h.T).max():.1e} (limit 1e-5)")
    assert a.meta["boundary_volume"][1] < 0 and b.meta["boundary_volume"][3] < 0
    assert a.meta["boundary_volume"][1] == pytest.approx(b.meta["boundary_volume"][3], rel=1e-5)
    assert np.abs(a.h - b.h.T).max() <= 1e-5
    assert np.abs(a.hu - b.hv.T).max() <= 1e-5


def test_level_condition_fills_basin(run):
    """A level above the dry bed of a closed basin fills it: the water enters through that side only, and the
    basin oscillates about the level (a seiche of a few % of the volume, period ~ 2 L / sqrt(g h) = 36 s, slowly
    damped by the friction: the level condition reflects the waves). Checked on the mean of the last 200 s."""
    c = swetest.pool("fill", 40, 20, 1.0, h=0.0, bc={"west": ["level", 0.5]}, manning=0.05)
    r = run(c, time=600.0, record_every=2.0)
    assert_mass(r, tol=1e-6)
    assert r.meta["boundary_volume"][0] == pytest.approx(r.meta["boundary_total"])
    t, V = swepost.volume_series(r.rec)
    r.plot.lines("volume", t, [("V(t)", V, "num"), ("target: level x area", np.full_like(t, 0.5 * 40 * 20), "ref")],
                 xlabel="t (s)", ylabel="volume (m3)", title="Basin filled through a level condition (seiche around the target)",
                 vlines=[(t[-1] - 200.0, "mean taken after")], note="limit: mean of the last 200 s within 1 % of the target")
    late = t >= t[-1] - 200.0
    target = 0.5 * 40 * 20
    assert abs(V[late].mean() - target) <= 1e-2 * target, f"mean volume {V[late].mean():.2f} m3, level gives {target}"
    assert np.abs(r.h - 0.5).max() <= 5e-2


def test_face_group_equals_domain_side(run):
    """A weak inflow condition on the west faces of the first column behaves like the west side of the domain."""
    nx, ny, q = 60, 4, 1.0
    side = swetest.channel("side_in", nx, ny, 1.0, slope=1e-3, h=0.3, manning=0.03,
                           bc={"west": ["inflow", q * ny], "east": ["free"]})
    faces = [(0 * ny + j) * 4 + 0 for j in range(ny)]
    group = swetest.channel("group_in", nx, ny, 1.0, slope=1e-3, h=0.3, manning=0.03, bc={"east": ["free"]},
                            face_bc=[{"kind": "inflow", "value": q * ny, "label": "in", "faces": faces}])
    a = run(side, time=100.0, name="side")
    b = run(group, time=100.0, name="group")
    assert_mass(a, tol=1e-6)
    assert_mass(b, tol=1e-6)
    x = swetest.centres(nx, 1.0)
    a.plot.lines("profile", x, [("west side of the domain: h", a.h[:, 1], "num"), ("face group: h", b.h[:, 1], "ref"), ("hu x 1 (west side)", a.hu[:, 1], "aux")],
                 title="Weak inflow on faces = west side of the domain", ylabel="h (m) / hu (m2/s)",
                 note=f"max |dh| = {np.abs(a.h - b.h).max():.1e}, max |dhu| = {np.abs(a.hu - b.hu).max():.1e} (limit 1e-6 relative)")
    assert b.meta["boundary_volume"][4] == pytest.approx(a.meta["boundary_volume"][0], rel=1e-6)
    assert b.meta["boundary_volume"][0] == 0.0
    assert np.abs(a.h - b.h).max() <= 1e-6 * max(a.h.max(), 1.0)
    assert np.abs(a.hu - b.hu).max() <= 1e-6 * max(np.abs(a.hu).max(), 1.0)


def test_interior_face_outflow_group(run):
    """An interior face group with a lower imposed level (an outlet line in the middle of a pool): the water
    leaves through it, the cells across see a wall, and the group's volume closes the balance. (A "free" group on
    still water drains nothing: its ghost repeats the cell, at rest.)"""
    nx, ny = 40, 20
    faces = [(20 * ny + j) * 4 + 1 for j in range(5, 15)]        # east faces of cells (20, 5..14)
    c = swetest.pool("face_out", nx, ny, 1.0, h=0.5, face_bc=[{"kind": "level", "value": 0.2, "label": "out", "faces": faces}])
    r = run(c, time=60.0, record_every=2.0)
    assert_mass(r, tol=1e-6)
    r.plot.fields("final", [("final h (m)", r.h), ("final hu (m2/s)", r.hu)], extent=(0, nx, 0, ny),
                  title=f"Outlet line (x = 20, y = 5..15) at level 0.2 m in a pool at 0.5 m; group removed {-r.meta['boundary_volume'][4]:.1f} m3")
    assert r.meta["boundary_volume"][4] < 0
    assert r.meta["boundary_volume"][4] == pytest.approx(r.meta["boundary_total"])
