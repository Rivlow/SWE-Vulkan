"""Mass conservation, the first property every run must have.

wolfgpu counterparts: test_gpu_sim.py test_cube_drop, test_small_splash, test_sine_ground_sine_water_3lanes,
test_steps_3lanes, test_infiltration_over_time*, test_injection.py (injected volumes). Here the tolerances are
those of float32 round-off (observed 1e-9 to 3e-8 relative), not 1e-3 to 1e-4 as there.

Every closed run is also recorded: the volume of every record must stay at V0, not only the final one.
"""
import numpy as np
import pytest

import swetest
from swetest import assert_mass

SCHEMES = ["euler", "ssprk2", "ssprk3"]


# ---- closed domains (walls): V(t) = V0 --------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", SCHEMES)
def test_dam_break_closed(run, scheme):
    r = run("dam_break", time=2.0, scheme=scheme, record_every=0.1)
    assert_mass(r, tol=1e-6)


@pytest.mark.parametrize("passes", [1, 4, 8])
def test_drain_test_limiter(run, passes):
    """Dam break at CFL 0.9 (Euler): needs the drain limiter; no clamp, h >= 0, mass to round-off."""
    r = run("drain_test", steps=1000, drain=passes, record_every=0.05)
    assert_mass(r, tol=1e-6)


def test_drain_test_without_limiter_creates_mass(run):
    """Sanity check of the case: without the limiter h < 0 is clamped, which creates mass."""
    r = run("drain_test", steps=1000, drain=0)
    assert r.meta["clamp_cells"] > 0
    assert r.meta["volume"] > r.meta["volume0"]


def test_lake_at_rest_closed(run):
    r = run("lake_at_rest", steps=1000, record_every=0.5)
    assert_mass(r, tol=1e-6)


def test_cube_drop(run):
    """wolfgpu test_cube_drop: a column of water dropped in a dry pool, walls."""
    n = 118
    h = np.zeros((n, n))
    h[49:69, 49:69] = 2.0
    c = swetest.pool("cube_drop", n, n, 1.0, h=h)
    r = run(c, time=60.0, record_every=2.0)
    assert_mass(r, tol=1e-6, volume0=c.volume0())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_sine_bed_sine_water(run, scheme):
    """wolfgpu test_sine_ground_sine_water_3lanes: sinusoidal bed and free surface, wet and dry areas."""
    nx, ny, cell = 200, 3, 0.5
    x = swetest.centres(nx, cell)
    zb = np.repeat((0.5 + 0.5 * np.sin(2 * np.pi * x / 25.0))[:, None], ny, axis=1)
    eta = 1.0 + 0.3 * np.sin(2 * np.pi * x / 40.0)
    h = np.maximum(eta[:, None] - zb, 0.0)
    c = swetest.pool("sine", nx, ny, cell, zb=zb, h=h, scheme=scheme)
    r = run(c, time=30.0, record_every=1.0)
    assert_mass(r, tol=1e-6, volume0=c.volume0())


def test_rocky_bed_splash(run):
    """Random bed, a raised block of water: fronts over rough dry ground."""
    n = 96
    zb = swetest.rocky_bed(n, n, 0.5, seed=3)
    h = np.maximum(0.6 - zb, 0.0)
    h[30:50, 30:50] = np.maximum(2.0 - zb[30:50, 30:50], 0.0)
    c = swetest.pool("rocky_splash", n, n, 1.0, zb=zb, h=h)
    r = run(c, time=40.0, record_every=2.0)
    assert_mass(r, tol=1e-6, volume0=c.volume0())


# ---- open domains: V(t) = V0 + integral of the imposed discharges ------------------------------------------------

def test_inflow_dry_domain(run):
    """5 m3/s through the west side of a dry domain: the boundary adds exactly 5 t (mass flux = imposed
    discharge), and the volume follows."""
    r = run("inflow_test", steps=3000, record_every=0.2)
    assert_mass(r, tol=1e-6, expected_added=lambda t: 5.0 * t)
    assert r.meta["boundary_volume"][0] > 0 and all(v == 0.0 for v in r.meta["boundary_volume"][1:4])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_source_zone(run, scheme):
    """A source zone adds Q t exactly."""
    q = 0.2
    c = swetest.pool("source_zone", 64, 64, 0.5, h=0.3, scheme=scheme,
                     zones=[{"kind": "source", "value": q, "label": "src", "rect": [10, 10, 13, 13]}])
    r = run(c, time=20.0, record_every=1.0)
    assert_mass(r, tol=1e-6, expected_added=lambda t: q * t)
    assert r.meta["zone_volume"][0] == pytest.approx(r.meta["boundary_total"])


def test_sink_and_level_zones(run):
    """A sink zone in a dam break and a level zone: the volume they remove / add closes the balance."""
    r = run("dam_break", steps=600, record_every=0.1,
            extra=["--zone", "sink,0,200,100,210,110", "--zone", "level,0.3,230,230,240,240"])
    assert_mass(r, tol=1e-6)
    assert r.meta["zone_volume"][0] < 0 and r.meta["zone_volume"][1] > 0


def test_face_inflow_group(run):
    """Weak inflow condition on interior faces (WOLF-style face group): the group adds Q t exactly."""
    nx, ny = 80, 40
    q = 2.0
    faces = [(10 * ny + j) * 4 + 0 for j in range(15, 25)]        # west faces of cells (10, 15..24)
    c = swetest.pool("face_inflow", nx, ny, 0.5, h=0.0, face_bc=[{"kind": "inflow", "value": q, "label": "in", "faces": faces}])
    r = run(c, time=10.0, record_every=0.5)
    assert_mass(r, tol=1e-6, expected_added=lambda t: q * t)
    assert r.meta["boundary_volume"][4] == pytest.approx(r.meta["boundary_total"])


@pytest.mark.parametrize("scheme", SCHEMES)
def test_inflow_outflow_balance(run, scheme):
    """Channel with an imposed discharge upstream and a free outflow downstream: V - V0 = in - out at every record."""
    nx, ny, cell, q = 100, 4, 1.0, 1.0
    c = swetest.channel("in_out", nx, ny, cell, slope=1e-3, h=0.5, manning=0.03, scheme=scheme,
                        bc={"west": ["inflow", q * ny * cell], "east": ["free"]})
    r = run(c, time=300.0, record_every=1.0)
    assert_mass(r, tol=1e-6)
    west, east = r.meta["boundary_volume"][0], r.meta["boundary_volume"][1]
    assert west == pytest.approx(q * ny * cell * r.t, rel=1e-5)
    assert east < 0


# ---- tiles: the active-tile scheme must give the same bits as the full grid ----------------------------------------

@pytest.mark.parametrize("args", [
    ("inflow_test", ["--scheme", "ssprk3", "--steps", 2000]),
    ("vesdre", ["--steps", 300]),
    ("dam_break", ["--scheme", "euler", "--steps", 600, "--water", "200,16,215,31,0.5", "--zone", "source,0.2,20,200,24,204",
                   "--zone", "level,0.3,230,230,240,240"]),
], ids=["inflow_test", "vesdre", "dam_break_blocks_zones"])
def test_active_tiles_bitwise(run, args):
    case, extra = args
    a = run(case, extra=extra, name="active")
    b = run(case, extra=extra, all_tiles=True, name="all")
    assert np.array_equal(a.Q.view(np.uint32), b.Q.view(np.uint32))


# ---- groundwater and chronologies: surface + aquifer - inputs is conserved ----------------------------------------

def test_rain_volume(run):
    """rain_infiltration: the rain table gives 2.9167 m3 (two zones, interpolated lines). The rates are read at the
    start of each step, so the rain volume differs from the integral of the table by O(dt) (observed 1.6e-4
    relative); what the surface received is checked exactly by test_surface_plus_aquifer."""
    r = run("rain_infiltration", steps=20000)
    swetest.assert_positive(r)
    assert abs(r.meta["rain_volume"] - 35.0 / 12.0) <= 1e-3 * 35.0 / 12.0


@pytest.mark.parametrize("case,extra", [
    ("rain_infiltration", ["--steps", 20000]),
    ("gw_seepage", ["--water", "100,100,140,140,0.5", "--steps", 20000]),
    ("gw_wolfgpu", ["--steps", 3000]),
])
def test_surface_plus_aquifer(run, case, extra):
    r = run(case, extra=extra)
    swetest.assert_positive(r)
    m = r.meta
    total0 = m["volume0"] + m.get("gw_volume0", 0.0)
    total = m["volume"] + m.get("gw_volume", 0.0) - m.get("sources_net", 0.0)
    assert abs(total - total0) <= 1e-5 * max(total0, 1e-30), f"{total0} -> {total}"
    if r.gw is not None:
        assert r.gw[..., 0].min() >= 0.0
