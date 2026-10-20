"""Lake at rest, exactly: still water over any bed must stay still to the bit, with every scheme, wet / dry
fronts, friction, walls and boundary conditions (wolfgpu test_still_water_rocky_bottom, *_increment_fens,
*_increment_alpine, test_lakeatrest_rockybottom_conservative_vs_nonconservative, test_v_shape_pool_and_no_flow,
test_hmod_hold_water, test_channel_wo_bridge_underpressure...).

The solver is balanced in floating point (see solver.slang balanced_source): when h + zb gives the same float32
level in every wet cell, the discharges stay exactly 0 and the depths exactly unchanged. wolfgpu tolerates
|q| < 1e-5 (conservative pressure) or sum |q| < 2e-4; here the criterion is equality to the bit.
The initial depths are built with swetest.depth_for_level (h + zb == level in float32), as the solver does for
the cases, the "niveau" edits and the level zones.
"""
import numpy as np
import pytest

import swetest
from swetest import assert_mass

SCHEMES = ["euler", "ssprk2", "ssprk3"]


def assert_still(r, h0):
    """Same depths to the bit, zero discharges to the bit, mass."""
    assert_mass(r, tol=0.0) if r.meta.get("boundary_total", 0.0) == 0.0 else assert_mass(r, tol=1e-9)
    moved = np.abs(r.Q[..., 1:]).max()
    dh_all = np.abs(r.Q[..., 0].astype(np.float64) - np.asarray(h0, dtype=np.float32).astype(np.float64))
    r.plot.bars("still_water", [("max |q| (spurious current)", moved, None), ("max |h - h0|", dh_all.max(), None)],
                title="Lake at rest: what moved (criterion: exactly 0, drawn at the 1e-18 floor)", ylabel="m2/s / m",
                note="wolfgpu tolerates |q| < 1e-5 or sum|q| < 2e-4; here the bits must be identical")
    r.plot.fields("setup", [("bed zb (m)", r.zb), ("initial depth h0 (m)", np.asarray(h0, dtype=np.float64)),
                            ("final |q| (m2/s)", np.hypot(r.hu, r.hv))], title="Lake at rest: bed, water and final discharge (all zero)")
    assert moved == 0.0, f"spurious discharge up to {moved:.3e} m2/s"
    dh = np.abs(r.Q[..., 0].astype(np.float64) - np.asarray(h0, dtype=np.float32).astype(np.float64)).max()
    assert dh == 0.0, f"depth changed by up to {dh:.3e} m"


def _rocky(name, n, amplitude, level, offset=0.0, seed=1, **kw):
    zb = (swetest.rocky_bed(n, n, amplitude, seed=seed) + offset).astype(np.float32)
    h = swetest.depth_for_level(level + offset, zb)
    return swetest.pool(name, n, n, 1.0, zb=zb, h=h, **kw), h


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("offset", [0.0, 200.0, 1000.0], ids=["z0", "fens_200m", "alpine_1000m"])
def test_rocky_bed_submerged(run, scheme, offset):
    """Random bed in [0, 2.5] m under 5 m of water (wolfgpu: 1000 steps), absolute elevations up to 1000 m
    (z_ref = 0: float32 has only 6e-5 m of resolution there, the lake must still be exact)."""
    c, h0 = _rocky(f"rocky_{int(offset)}_{scheme}", 100, 2.5, 5.0, offset, scheme=scheme)
    r = run(c, steps=1000, record_every=5.0)
    assert_still(r, h0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_rocky_bed_wolfgpu_setup(run, scheme):
    """wolfgpu test_lakeatrest_rockybottom_conservative_vs_nonconservative: bed U[0, 0.5], eta = 1, CFL 0.2."""
    c, h0 = _rocky(f"rocky_wolfgpu_{scheme}", 64, 0.5, 1.0, seed=42, scheme=scheme, cfl=0.2)
    r = run(c, steps=1000)
    assert_still(r, h0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_rocky_bed_with_islands(run, scheme):
    """Random bed in [0, 2.5] m under eta = 1.5 m: a third of the cells are dry islands, wet/dry faces
    everywhere, Manning friction."""
    c, h0 = _rocky(f"rocky_islands_{scheme}", 100, 2.5, 1.5, scheme=scheme, manning=0.04)
    assert (h0 == 0).mean() > 0.2
    r = run(c, steps=2000, record_every=5.0)
    assert_still(r, h0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_bump_island(run, scheme):
    """lake_at_rest: eta = 0.5 over the analytical bump, a dry island at its top."""
    r = run("lake_at_rest", steps=1000, scheme=scheme, record_every=0.5, name="lake")
    ref = run("lake_at_rest", steps=0, name="initial")
    assert_still(r, ref.Q[..., 0])


def test_v_shape_with_friction(run):
    """V-shaped valley, Manning 0.04, eta = 0.5 with wet and dry banks."""
    nx, ny = 40, 40
    j = swetest.centres(ny, 1.0)
    zb = np.repeat((np.abs(j - 20.0) / 20.0)[None, :], nx, axis=0).astype(np.float32)
    h0 = swetest.depth_for_level(0.5, zb)
    c = swetest.pool("v_shape", nx, ny, 1.0, zb=zb, h=h0, manning=0.04)
    r = run(c, steps=2000)
    assert_still(r, h0)


@pytest.mark.parametrize("kind", ["level", "depth"])
def test_conditions_hold_water(run, kind):
    """wolfgpu test_hmod_hold_water: rocky bed, every side with the still-water level (or, on a flat bed, the depth)
    imposed: nothing moves, nothing enters or leaves."""
    if kind == "level":
        c, h0 = _rocky("hold_level", 48, 0.8, 1.0, bc={s: ["level", 1.0] for s in ("west", "east", "south", "north")})
    else:
        h0 = np.full((48, 48), 1.0, dtype=np.float32)
        c = swetest.pool("hold_depth", 48, 48, 1.0, h=h0, bc={s: ["depth", 1.0] for s in ("west", "east", "south", "north")})
    r = run(c, steps=2000)
    assert_still(r, h0)
    assert all(v == 0.0 for v in r.meta["boundary_volume"])


def test_interior_wall_group(run):
    """A line of interior walls (weak face condition) in a lake on a rough bed: the walls hold still water too."""
    n = 48
    c, h0 = _rocky("wall_group", n, 0.8, 1.0)
    faces = [(20 * n + j) * 4 + 1 for j in range(10, 38)]
    c.face_bc = [{"kind": "wall", "value": 0.0, "label": "digue", "faces": faces}]
    r = run(c, steps=2000)
    assert_still(r, h0)


def test_level_zone_and_level_edit(run):
    """A level zone at the lake's level and the "niveau" edit (--water applies a depth; the level edit is
    exercised through a level zone equal to the surrounding lake): still."""
    n = 48
    c, h0 = _rocky("level_zone", n, 0.8, 1.0, zones=[{"kind": "level", "value": 1.0, "label": "lac", "rect": [5, 5, 10, 10]}])
    r = run(c, steps=2000)
    assert_still(r, h0)
    assert r.meta["zone_volume"][0] == 0.0
