"""Bridges: free surface under the deck, pressurised flow once the water reaches the soffit, overtopping above the
deck top (shaders/solver/bridge.slang). Inspired by wolfgpu tests/test_bridges.py (50 x 10 channel, dx = 1 m,
uniform deck, discharge upstream, depth downstream), which only covers the pressurised regime (the deck top is never
read by wolfgpu: no overtopping) and checks against its own saved results; here every test closes the mass balance
on the physical (stored) volume, and the steady states are checked against hydraulics:
continuity, rest, friction slope under pressure, overtopping.
"""
import numpy as np
import pytest

import swepost
import swetest
from swetest import assert_mass

NX, NY, CELL = 50, 10, 1.0
BRIDGE = [20, 0, 29, NY - 1]        # i0, j0, i1, j1: the deck spans the whole width


def channel(name, h, q=0.0, roof=5.0, deck=6.0, manning=0.0, bed=None, scheme="ssprk2", up=None, down=None, slot=0.05,
            bridges=None):
    zb = np.zeros((NX, NY)) if bed is None else bed
    eta = np.float32(h) if np.ndim(h) == 0 else None
    h0 = swetest.depth_for_level(eta, zb) if eta is not None else np.asarray(h, dtype=np.float32)
    bc = dict(swetest.WALLS)
    if up is not None:
        bc["west"] = ["inflow", up]
    if down is not None:
        bc["east"] = ["depth", down]
    c = swetest.Case(name, CELL, zb, h0, np.where(h0 > 0, q, 0.0) if q else None, None, bc=bc, manning=manning, scheme=scheme,
                     bridge_slot=slot,
                     bridges=bridges if bridges is not None else [{"label": "pont", "z_roof": roof, "z_deck": deck, "rect": BRIDGE}])
    return c


def under(r, i0=BRIDGE[0], i1=BRIDGE[2]):
    return slice(i0, i1 + 1)


def assert_continuity(r, q):
    """Steady flow: every cell carries the discharge q, within 0.2 % away from the deck's entrance and exit, 1 % on
    the two cells of each transition (the cell-centred discharge is not the face flux where the opening changes;
    observed 0.4 %)."""
    dev = np.abs(r.hu[2:-2, :] - q).max(axis=1)
    i = np.arange(2, NX - 2)
    near = (np.abs(i - BRIDGE[0]) <= 2) | (np.abs(i - BRIDGE[2]) <= 2)
    assert dev[~near].max() <= 2e-3 * q, f"discharge off by {dev[~near].max():.4f} m2/s away from the deck ends"
    assert dev.max() <= 1e-2 * q, f"discharge off by {dev.max():.4f} m2/s at the deck ends"


# ---- at rest: exactly, in every regime -------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["euler", "ssprk2", "ssprk3"])
@pytest.mark.parametrize("level,regime", [(3.0, "free"), (5.6, "pressurised"), (7.5, "overtopping")])
def test_lake_at_rest_under_bridge(run, scheme, level, regime):
    """wolfgpu test_channel_wo_bridge_noflow / _underpressure / _underpressure_float: still water stays exactly still,
    whether it stays below the soffit, fills the opening under pressure, or covers the deck."""
    c = channel(f"rest_{regime}_{scheme}", level, scheme=scheme)
    r = run(c, steps=2000, record_every=2.0)
    assert_mass(r, tol=1e-12, stored=r.stored())        # host and post-processing sum in different orders
    assert np.abs(r.Q[..., 1:]).max() == 0.0
    assert np.array_equal(r.Q[..., 0], c.h0.astype(np.float32))


def test_lake_at_rest_sloped_bed_triangular_soffit(run):
    """wolfgpu test_channel_wo_bridge_vshape_bedslope_5000stepsRK: bed sloping 12 -> 5 m, eta = 16 m, soffit varying
    across the channel between 14 and 15 m, deck top 15.5 m: partly pressurised, everything at rest."""
    x = swetest.centres(NX, CELL)
    bed = np.repeat((12.0 - 7.0 * x / (NX * CELL))[:, None], NY, axis=1)
    bridges = []
    for j in range(NY):
        z = 14.0 + abs(j - (NY - 1) / 2) / ((NY - 1) / 2)
        bridges.append({"label": f"arc {j}", "z_roof": z, "z_deck": 15.5 if z < 15.5 else z, "rect": [BRIDGE[0], j, BRIDGE[2], j]})
    c = channel("rest_vshape", 16.0, bed=bed, bridges=bridges)
    r = run(c, steps=3000)
    assert_mass(r, tol=1e-12, stored=r.stored())        # host and post-processing sum in different orders
    assert np.abs(r.Q[..., 1:]).max() == 0.0


# ---- pressurised flow ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("manning", [0.0, 0.04])
def test_pressurised_flow(run, manning):
    """wolfgpu test_channel_wo_bridge_flowing_underpressure_mann0 / _mann004: q = 5 m2/s, h = 6 m, soffit at 5 m,
    discharge upstream, depth 6 m downstream. Steady state: the same discharge everywhere (continuity, exact outflow
    rate), the opening runs full (h > c), and under the deck the piezometric line falls with the friction slope of
    a pressurised flow (R = c / 2: bed and soffit wetted)."""
    q = 5.0
    c = channel(f"pressure_{manning}", 6.0, q=q, roof=5.0, deck=6.5, manning=manning, up=q * NY * CELL, down=6.0)
    r = run(c, time=600.0, record_every=50.0)
    assert_mass(r, tol=1e-5, stored=r.stored())
    h = r.h[:, NY // 2]
    assert np.all(h[under(r)] > 5.0), "the opening must run full"
    # continuity: every cell carries q (steady), and the east side lets out q * width
    assert_continuity(r, q)
    b = swepost.exact_balance(r.rec)
    east = [p["east"] for p in b["parts"]]
    q_out = -(east[-1] - east[-2]) / (b["t"][-1] - b["t"][-2])
    assert abs(q_out - q * NY * CELL) <= 1e-3 * q * NY * CELL
    if manning > 0:
        # piezometric slope inside the bridge (away from its entrance and exit cells) = pressurised friction slope
        eta = r.eta[:, NY // 2]
        i = np.arange(BRIDGE[0] + 2, BRIDGE[2] - 1)
        slope = -np.polyfit(i * CELL, eta[i], 1)[0]
        sf = swetest.pressurised_friction_slope(q, manning, 5.0)
        assert abs(slope - sf) <= 0.1 * sf, f"piezometric slope {slope:.3e}, friction slope {sf:.3e}"
    else:
        # frictionless: the energy head is conserved through the contraction up to the expansion loss downstream;
        # under the deck the faster flow (u = q / c) lowers the piezometric head by about (u_b^2 - u_0^2) / 2g
        eta = r.eta[:, NY // 2]
        drop = eta[BRIDGE[0] - 5] - eta[(BRIDGE[0] + BRIDGE[2]) // 2]
        bernoulli = ((q / 5.0) ** 2 - (q / eta[BRIDGE[0] - 5]) ** 2) / (2 * swetest.G)
        assert 0.5 * bernoulli <= drop <= 2.0 * bernoulli, f"head drop under the deck {drop:.4f} m, Bernoulli {bernoulli:.4f} m"


# ---- overtopping ---------------------------------------------------------------------------------------------------

def test_overtopping(run):
    """Downstream level above the deck top: water flows under the deck (pressurised) and over it. Continuity holds,
    the cells over the deck carry the overtopping layer, and the mass closes on the stored volume."""
    q = 3.0
    c = channel("overtop", 7.0, q=q, roof=5.0, deck=6.0, manning=0.03, up=q * NY * CELL, down=7.0)
    r = run(c, time=600.0, record_every=50.0)
    assert_mass(r, tol=1e-5, stored=r.stored())
    h = r.h[:, NY // 2]
    assert np.all(h[under(r)] > 6.0), "the water must cover the deck"
    assert_continuity(r, q)


@pytest.mark.slow
def test_rising_flood_through_all_regimes(run):
    """A discharge pushed into a closed reach with a bridge: the water rises from below the soffit to above the deck
    (free -> pressurised -> overtopping); the mass closes at every record (stored volume = what entered)."""
    q = 2.0
    c = channel("rising", 2.0, roof=4.0, deck=5.0, manning=0.03, up=q * NY * CELL)
    r = run(c, time=1500.0, record_every=25.0)
    assert_mass(r, tol=1e-6, stored=r.stored(), expected_added=lambda t: q * NY * CELL * t)
    assert r.h[under(r), NY // 2].max() > 5.0, "the flood must reach the overtopping regime"
    b = swepost.exact_balance(r.rec, r.stored())
    assert np.abs(b["residual_rel"]).max() <= 1e-6


def test_slot_one_matches_storage_of_piezometric_column(run):
    """bridge_slot = 1: no Preissmann slot, the pressurised cell stores its whole piezometric column (wolfgpu's
    model): the stored volume is then sum(h) and the mass closes on it."""
    q = 2.0
    c = channel("slot1", 2.0, roof=4.0, deck=5.0, manning=0.03, up=q * NY * CELL, slot=1.0)
    r = run(c, time=600.0)
    assert_mass(r, tol=1e-6, stored=r.stored(), expected_added=lambda t: q * NY * CELL * t)
    assert r.volume(r.stored()) == pytest.approx(r.volume(), rel=1e-6)


@pytest.mark.parametrize("manning", [0.0, 0.04])
def test_against_wolfgpu_reference(run, wolfgpu, manning):
    """wolfgpu tests/test_bridges.py test_channel_w_bridge_horizontal_flowing_underpressure_mann0 / _mann004: its
    saved steady profiles (tests/expected_results/test_channel_w_bridge_man0*). Same geometry: 48 computed cells (15
    upstream, a 20 m deck with its soffit at 5 m, 13 downstream), q = 5 m2/s, depth 6 m downstream, flat bed.
    wolfgpu has no slot (bridge_slot = 1 here). Compared: the discharge, and the piezometric drop at the deck
    entrance (Bernoulli 15.6 mm; wolfgpu 17.1 mm). Not compared: the slope under the deck with friction - wolfgpu
    evaluates Manning with the whole piezometric depth (7.4e-4), here the hydraulic radius of the full opening
    R = c / 2 (4.7e-4, the friction slope of a pressurised conduit)."""
    repo, _ = wolfgpu
    tag = "man0" if manning == 0.0 else "man004"
    ref_h = np.load(repo / "tests" / "expected_results" / f"test_channel_w_bridge_{tag}_h.npy")[1:-1]
    ref_q = np.load(repo / "tests" / "expected_results" / f"test_channel_w_bridge_{tag}_qx.npy")[1:-1]
    nx, ny, q = ref_h.size, 6, 5.0
    zb = np.zeros((nx, ny))
    h0 = np.full((nx, ny), 6.0, dtype=np.float32)
    bc = dict(swetest.WALLS)
    bc["west"] = ["inflow", q * ny]
    bc["east"] = ["depth", 6.0]
    c = swetest.Case(f"wolfgpu_{tag}", 1.0, zb, h0, np.full((nx, ny), q), None, bc=bc, manning=manning, bridge_slot=1.0,
                     bridges=[{"label": "pont", "z_roof": 5.0, "z_deck": 6.5, "rect": [15, 0, 34, ny - 1]}])
    r = run(c, time=600.0)
    assert_mass(r, tol=1e-5, stored=r.stored())
    h, hu = r.h[:, ny // 2], r.hu[:, ny // 2]
    assert np.abs(hu[2:-2] - ref_q[2:-2]).max() <= 2e-2 * q
    drop, ref_drop = h[12] - h[17], ref_h[12] - ref_h[17]
    assert abs(drop - ref_drop) <= 0.15 * ref_drop, f"entrance drop {drop * 1e3:.1f} mm, wolfgpu {ref_drop * 1e3:.1f} mm"
    assert np.abs(h - ref_h).max() <= 2e-2, f"piezometric depth differs from wolfgpu by {np.abs(h - ref_h).max():.4f} m"


def test_weir_deck_on_the_bed(run):
    """A soffit on the bed makes the deck a weir: nothing passes until the water tops it, then the flow over the
    crest is controlled: q = Cd sqrt(g) (2/3 E)^1.5 for an upstream head E above the crest. Ideal broad-crested weir
    Cd = 1 (critical flow, no loss): the upstream head can only be higher (no energy gain), and a short crest (2 m
    here) gives Cd ~ 0.85-0.95 in practice; observed Cd = 0.86."""
    q = 1.0
    c = channel("weir", 1.5, roof=0.0, deck=1.0, manning=0.0, up=q * NY * CELL, bridges=[
        {"label": "seuil", "z_roof": 0.0, "z_deck": 1.0, "rect": [24, 0, 25, NY - 1]}])
    c.bc["east"] = ["free"]
    r = run(c, time=800.0, record_every=50.0)
    assert_mass(r, tol=1e-5, stored=r.stored())
    eta_up = r.eta[15, NY // 2]
    head = eta_up - 1.0 + (q / r.h[15, NY // 2]) ** 2 / (2 * swetest.G)
    q_weir = np.sqrt(swetest.G) * (2.0 / 3.0 * head) ** 1.5        # ideal discharge for that head
    cd = q / q_weir
    assert 0.8 <= cd <= 1.0, f"upstream head {head:.3f} m: Cd = {cd:.3f}"
    # nothing below the crest passes: the weir cells hold no water under the deck (soffit on the bed)
    assert r.meta["boundary_volume"][1] < 0        # and it does flow out downstream
