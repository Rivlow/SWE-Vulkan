"""Helpers of the hydraulic tests: generated cases, analytical solutions, assertions.

A `Case` is written as an "imported" case (JSON + npz) in a temporary assets folder and run by `swe --headless`
(fixture `run` of conftest.py). Arrays are (nx, ny) in numpy C order like the solver (i along x, j along y).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

G = 9.81

WALLS = {"west": ["wall"], "east": ["wall"], "south": ["wall"], "north": ["wall"]}


# ---------------------------------------------------------------------------------------------------------------
# Generated cases
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    cell: float
    zb: np.ndarray                     # (nx, ny), relative to z_ref
    h0: np.ndarray
    hu0: np.ndarray | None = None
    hv0: np.ndarray | None = None
    z_ref: float = 0.0
    bc: dict = field(default_factory=lambda: dict(WALLS))
    manning: float = 0.0
    scheme: str = "ssprk2"
    cfl: float = 0.4
    drain_passes: int = 4
    label: str = ""
    zones: list = field(default_factory=list)          # {"kind", "value", "label", "rect": [i0, j0, i1, j1]}
    face_bc: list = field(default_factory=list)
    bridges: list = field(default_factory=list)        # {"label", "z_roof", "z_deck", "rect": [i0, j0, i1, j1]}
    bridge_slot: float = 0.05
    extra: dict = field(default_factory=dict)          # merged into the JSON (groundwater, rain, infiltration...)

    @property
    def nx(self):
        return self.zb.shape[0]

    @property
    def ny(self):
        return self.zb.shape[1]

    @property
    def area(self):
        return self.cell * self.cell

    def volume0(self):
        return float(self.h0.astype(np.float64).sum() * self.area)

    def bridge_arrays(self):
        """Per-cell soffit and deck elevations (NaN = no bridge), from the rectangles."""
        roof = np.full(self.zb.shape, np.nan)
        deck = np.full(self.zb.shape, np.nan)
        for b in self.bridges:
            i0, j0, i1, j1 = b["rect"]
            roof[i0:i1 + 1, j0:j1 + 1] = b["z_roof"]
            deck[i0:i1 + 1, j0:j1 + 1] = max(b["z_deck"], b["z_roof"])
        # the solver keeps them in float32
        return roof.astype(np.float32).astype(np.float64), deck.astype(np.float32).astype(np.float64)

    def write(self, assets):
        assets = Path(assets)
        (assets / "cases").mkdir(parents=True, exist_ok=True)
        (assets / "data").mkdir(parents=True, exist_ok=True)
        arrays = {"zb": self.zb.astype(np.float32), "h0": self.h0.astype(np.float32)}
        if self.hu0 is not None:
            arrays["hu0"] = self.hu0.astype(np.float32)
        if self.hv0 is not None:
            arrays["hv0"] = self.hv0.astype(np.float32)
        if self.bridges:
            roof, deck = self.bridge_arrays()
            arrays["bridge_roof"] = roof.astype(np.float32)
            arrays["bridge_deck"] = deck.astype(np.float32)
        np.savez(assets / "data" / f"{self.name}.npz", **arrays)
        j = {"label": self.label or self.name, "type": "imported", "data": f"data/{self.name}.npz", "cell": self.cell,
             "z_ref": self.z_ref, "eta_0": 0.0, "manning_n": self.manning, "cfl": self.cfl, "scheme": self.scheme,
             "drain_passes": self.drain_passes, "bc": self.bc, "render": {"depth_ref": 1.0, "z_scale_0": 1.0}}
        if self.zones:
            j["zones"] = self.zones
        if self.face_bc:
            j["face_bc"] = self.face_bc
        if self.bridges:
            j["bridges"] = self.bridges
            j["bridge_slot"] = self.bridge_slot
        j.update(self.extra)
        (assets / "cases" / f"{self.name}.json").write_text(json.dumps(j, indent=1), encoding="utf-8")
        return assets


def centres(n, cell):
    return (np.arange(n) + 0.5) * cell


def channel(name, nx, ny, cell, slope=0.0, h=1.0, q=0.0, manning=0.0, bc=None, z0=None, **kw):
    """Rectangular channel along x: bed falling with `slope` (z = z0 - slope x, z0 so that the lowest bed is 0),
    uniform depth h and unit discharge q (hu = q), walls on the sides."""
    x = centres(nx, cell)
    if z0 is None:
        z0 = slope * nx * cell
    zb = np.repeat((z0 - slope * x)[:, None], ny, axis=1)
    h0 = np.full((nx, ny), float(h))
    hu0 = np.full((nx, ny), float(q)) if q else None
    bcd = dict(WALLS)
    if bc:
        bcd.update(bc)
    return Case(name, cell, zb, h0, hu0, None, bc=bcd, manning=manning, **kw)


def pool(name, nx, ny, cell, zb=None, h=None, **kw):
    """Closed pool: flat bed (or `zb`) and still water of depth h (or the array h)."""
    if zb is None:
        zb = np.zeros((nx, ny))
    if h is None:
        h0 = np.zeros((nx, ny))
    else:
        h0 = np.full((nx, ny), float(h)) if np.ndim(h) == 0 else np.asarray(h, dtype=np.float64)
    return Case(name, cell, np.asarray(zb, dtype=np.float64), h0, **kw)


def depth_for_level(level, zb):
    """float32 depth of still water at `level` such that h + zb == level exactly in float32 (as the solver's
    depth_for_level: the lake is then flat to the bit), 0 where the bed is above the level."""
    z = np.asarray(zb, dtype=np.float32)
    L = np.float32(level)
    h = (L - z).astype(np.float32)
    for _ in range(4):
        s = (h + z).astype(np.float32)
        bad = (s != L) & (h > 0)
        if not bad.any():
            break
        h = np.where(bad, np.nextafter(h, np.where(s < L, np.float32(np.inf), np.float32(0))), h).astype(np.float32)
    return np.where(h > 0, h, np.float32(0)).astype(np.float32)


def rocky_bed(nx, ny, amplitude, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nx, ny)) * amplitude


def bump_bed(nx, cell, ny=1, length=25.0):
    """SWASHES bump: zb = max(0, 0.2 - 0.05 (x - 10)^2) on [0, 25] m (nx cells of `cell`)."""
    x = centres(nx, cell)
    z = np.maximum(0.0, 0.2 - 0.05 * (x - 10.0) ** 2)
    return np.repeat(z[:, None], ny, axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Analytical solutions (1-D along x; x measured from the dam for the dam breaks)
# ---------------------------------------------------------------------------------------------------------------

def ritter(x, t, h_l, x0=0.0):
    """Dam break on a dry bed: h(x, t) and u(x, t)."""
    c0 = np.sqrt(G * h_l)
    xi = (x - x0) / max(t, 1e-30)
    h = np.where(xi <= -c0, h_l, np.where(xi <= 2 * c0, (2 * c0 - xi) ** 2 / (9 * G), 0.0))
    u = np.where(xi <= -c0, 0.0, np.where(xi <= 2 * c0, 2.0 / 3.0 * (xi + c0), 0.0))
    return h, u


def stoker(x, t, h_l, h_r, x0=0.0):
    """Dam break on a wet bed (Stoker): rarefaction, constant middle state, shock. Returns h, u."""
    c_l, c_r = np.sqrt(G * h_l), np.sqrt(G * h_r)

    def f(c_m):
        return -8 * G * h_r * c_m ** 2 * (c_l - c_m) ** 2 + (c_m ** 2 - G * h_r) ** 2 * (c_m ** 2 + G * h_r)

    c_m = brentq(f, c_r * (1 + 1e-9), c_l * (1 - 1e-9))
    h_m = c_m ** 2 / G
    u_m = 2 * (c_l - c_m)
    s = 2 * c_m ** 2 * (c_l - c_m) / (c_m ** 2 - c_r ** 2)        # shock speed
    xi = (x - x0) / max(t, 1e-30)
    h = np.where(xi <= -c_l, h_l,
                 np.where(xi <= 2 * c_l - 3 * c_m, (2 * c_l - xi) ** 2 / (9 * G),
                          np.where(xi <= s, h_m, h_r)))
    u = np.where(xi <= -c_l, 0.0, np.where(xi <= 2 * c_l - 3 * c_m, 2.0 / 3.0 * (xi + c_l), np.where(xi <= s, u_m, 0.0)))
    return h, u


def normal_depth(q, n, slope):
    """Manning normal depth of a wide channel: h_n = (n q / sqrt(S))^(3/5)."""
    return (n * q / np.sqrt(slope)) ** 0.6


def bernoulli_depth(q, head, zb, supercritical=False):
    """Depth of the root of h + q^2 / (2 g h^2) + zb = head (subcritical or supercritical branch); NaN when none."""
    h_c = (q * q / G) ** (1.0 / 3.0)
    e_min = h_c + q * q / (2 * G * h_c * h_c)
    e = head - zb
    if e < e_min * (1 - 1e-12):
        return np.nan

    def f(h):
        return h + q * q / (2 * G * h * h) + zb - head

    if abs(e - e_min) < 1e-12:
        return h_c
    return brentq(f, 1e-9 * h_c, h_c) if supercritical else brentq(f, h_c, 10 * max(e, h_c))


def bump_subcritical(x, zb, q=4.42, h_out=2.0):
    """SWASHES 3.1.1: subcritical flow over the bump, outlet depth h_out, frictionless: Bernoulli from the outlet."""
    head = h_out + q * q / (2 * G * h_out * h_out)        # zb = 0 at the outlet
    return np.array([bernoulli_depth(q, head, z) for z in zb])


def bump_transcritical(x, zb, q=1.53):
    """SWASHES 3.1.2: transcritical without shock: critical at the crest, subcritical upstream, supercritical
    downstream (free outflow)."""
    h_c = (q * q / G) ** (1.0 / 3.0)
    head = h_c + q * q / (2 * G * h_c * h_c) + zb.max()
    i_c = int(np.argmax(zb))
    return np.array([bernoulli_depth(q, head, z, supercritical=i > i_c) for i, z in enumerate(zb)])


def bump_shock(x, zb, q=0.18, h_out=0.33):
    """SWASHES 3.1.3: transcritical with a hydraulic jump downstream of the crest. The jump position comes from
    the Rankine-Hugoniot relation between the supercritical profile (from the crest) and the subcritical profile
    (from the outlet). Returns h and the jump abscissa."""
    h_c = (q * q / G) ** (1.0 / 3.0)
    head_up = h_c + q * q / (2 * G * h_c * h_c) + zb.max()
    head_down = h_out + q * q / (2 * G * h_out * h_out)
    i_c = int(np.argmax(zb))
    h_sup = np.array([bernoulli_depth(q, head_up, z, supercritical=i > i_c) if i >= i_c else np.nan for i, z in enumerate(zb)])
    h_sub = np.array([bernoulli_depth(q, head_down, z) for z in zb])

    def momentum(h):
        return q * q / (G * h) + 0.5 * h * h

    jump = momentum(h_sup) - momentum(h_sub)        # changes sign at the jump
    cand = [i for i in range(i_c + 1, len(x) - 1) if np.isfinite(jump[i]) and np.isfinite(jump[i + 1]) and jump[i] * jump[i + 1] <= 0]
    i_s = cand[0] if cand else len(x) - 1
    x_s = x[i_s]
    h = np.where(np.arange(len(x)) <= i_s, np.array([bernoulli_depth(q, head_up, z, supercritical=i > i_c) for i, z in enumerate(zb)]), h_sub)
    return h, x_s


def backwater_m2(x, q, n, slope, h_down):
    """Gradually varied flow (wide channel): dh/dx = (S0 - Sf) / (1 - Fr^2) integrated upstream from the depth
    h_down at x[-1]. Returns h on x."""
    def rhs(s, h):
        sf = n * n * q * q / h ** (10.0 / 3.0)
        fr2 = q * q / (G * h ** 3)
        return (slope - sf) / (1 - fr2)

    sol = solve_ivp(rhs, (x[-1], x[0]), [h_down], t_eval=x[::-1], rtol=1e-10, atol=1e-12)
    return sol.y[0][::-1]


def pressurised_friction_slope(q, n, clearance):
    """Piezometric gradient of a uniform flow under a deck of clearance c (unit discharge q), Manning n with the
    hydraulic radius R = c / 2 (bed + soffit wetted)."""
    u = q / clearance
    R = clearance / 2.0
    return n * n * u * u / R ** (4.0 / 3.0)


# ---------------------------------------------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------------------------------------------

def rel_l1(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).sum() / max(np.abs(a).sum(), 1e-30))


def assert_positive(run):
    assert np.isfinite(run.Q).all(), "non-finite state"
    assert run.h.min() >= 0.0, f"negative depth {run.h.min():.3e}"
    assert run.meta.get("clamp_cells", 0) == 0, f"{run.meta['clamp_cells']} cell(s) clamped to h = 0 (max {run.meta['clamp_max']:.2e} m)"


def assert_mass(run, tol=1e-6, volume0=None, stored=None, expected_added=None, added_tol=1e-5, series_tol=None):
    """Mass conservation of a run, the first check of every test:
    - positivity: h >= 0, finite, no cell clamped to 0 (a clamp creates mass);
    - closure: V - V0 - added == 0 to `tol` (relative to the largest of V0, V, |added|), `added` being the volume
      the solver measured through every open side, face group and zone (meta.json boundary_total, exact);
    - when the run was recorded, the same closure on every record (`series_tol`, default `tol`);
    - `expected_added` (m3, a number or a function of t): what the conditions should have added (e.g. Q t for an
      imposed discharge), to `added_tol`: checks that the boundary delivers what is imposed.
    Returns the closure error."""
    import swepost

    assert_positive(run)
    if volume0 is None:
        volume0 = float(run.meta["volume0"])        # physical: stored depth under bridges (host, double)
    V = run.volume(stored)
    added = float(run.meta.get("added_total", run.meta.get("boundary_total", 0.0)))
    scale = max(abs(volume0), abs(V), abs(added), 1e-30)
    err = abs(V - volume0 - added) / scale
    if added != 0.0:
        # open domain: every step adds the boundary inflow to float32 depths, each update rounded (random walk of
        # about sqrt(N) float32 ulps of the volume); a closed domain stays far below that (1e-8 observed)
        tol = max(tol, 4.0 * np.sqrt(max(run.meta["steps"], 1)) * 2.0 ** -24)
        series_tol = max(series_tol or 0.0, tol)
    run.plot.mass(run, err, tol, volume0, V, added, stored, series_tol)
    assert err <= tol, (f"mass not conserved: V = {V:.9g} m3, V0 = {volume0:.9g}, added by the boundaries/zones "
                        f"{added:+.9g} -> closure error {V - volume0 - added:+.3e} m3 ({err:.3e} relative > {tol:.1e})")
    if expected_added is not None:
        want = expected_added(run.t) if callable(expected_added) else float(expected_added)
        e = abs(added - want) / max(abs(want), scale * 1e-3, 1e-30)
        assert e <= added_tol, f"boundaries added {added:.9g} m3, expected {want:.9g} ({e:.3e} relative > {added_tol:.1e})"
    # per record: the recordings carry the exact boundary / zone volumes only (chronologies and the aquifer are
    # balanced on the final state above)
    if run.rec is not None and "sources_net" not in run.meta and "gw_volume" not in run.meta:
        b = swepost.exact_balance(run.rec, stored)
        assert b is not None, "recording without the solver scalars (version 1)"
        res = np.abs(b["V"] - b["V"][0] - (b["added"] - b["added"][0])) / scale
        k = int(np.argmax(res))
        assert res.max() <= (series_tol or tol), (f"mass not conserved at t = {b['t'][k]:.4g} s (record {k}): closure "
                                                  f"{b['residual'][k]:+.3e} m3 ({res[k]:.3e} relative)")
    return err
