"""Uniform and steady flows with imposed discharge / depth (wolfgpu test_waterline_along_X_noflow_with_BC*,
test_still_flow_left_to_right / right_to_left / bottom_to_top / top_to_bottom, test_still_pool_with_qx_h_bc_*,
test_waterline_along_X_3lines_rough)."""
import numpy as np
import pytest

import swepost
import swetest
from swetest import assert_mass


def _uniform(name, axis, sign, n=104, w=3, h=5.0, q=1.0):
    """Flat frictionless channel along x or y, h and unit discharge sign * q everywhere; inflow on the upstream
    side, the depth h imposed downstream."""
    shape = (n, w) if axis == "x" else (w, n)
    zb = np.zeros(shape)
    hu = np.full(shape, sign * q) if axis == "x" else None
    hv = np.full(shape, sign * q) if axis == "y" else None
    up, down = (("west", "east") if sign > 0 else ("east", "west")) if axis == "x" else \
               (("south", "north") if sign > 0 else ("north", "south"))
    bc = dict(swetest.WALLS)
    bc[up] = ["inflow", q * w]
    bc[down] = ["depth", h]
    return swetest.Case(name, 1.0, zb, np.full(shape, h), hu, hv, bc=bc)


@pytest.mark.parametrize("axis,sign", [("x", 1), ("x", -1), ("y", 1), ("y", -1)],
                         ids=["left_to_right", "right_to_left", "bottom_to_top", "top_to_bottom"])
def test_uniform_flow_stays_uniform(run, axis, sign):
    c = _uniform(f"uniform_{axis}_{sign}", axis, sign)
    r = run(c, steps=2000, record_every=5.0)
    assert_mass(r, tol=1e-6, volume0=c.volume0())
    q_n = r.hu if axis == "x" else r.hv
    q_t = r.hv if axis == "x" else r.hu
    assert np.abs(r.h - 5.0).max() <= 1e-6
    assert np.abs(q_n - sign * 1.0).max() <= 1e-6
    assert np.abs(q_t).max() == 0.0


def test_uniform_flow_x_y_symmetry(run):
    """The same flow along x and along y gives the same field (transposed)."""
    a = run(_uniform("sym_x", "x", 1), steps=1000, name="x")
    b = run(_uniform("sym_y", "y", 1), steps=1000, name="y")
    assert np.abs(a.h - b.h.T).max() <= 1e-6
    assert np.abs(a.hu - b.hv.T).max() <= 1e-6


@pytest.mark.parametrize("q,n,slope", [(1.0, 0.03, 1e-3), (0.5, 0.05, 5e-3)])
def test_normal_depth(run, q, n, slope):
    """Sloping channel with Manning friction started at a wrong depth: converges to the normal depth
    h_n = (n q / sqrt(S))^(3/5) (wolfgpu test_waterline_along_X_3lines_rough, without assertion there).
    The depth h_n is imposed downstream on the ghost state, not on the cell: the last cell settles a few mm off, and
    that small backwater decays upstream over L = 3 h (1 - Fr^2) / (10 S) (260 m on the mild slope). The channel is
    2 km long and the upstream third is checked. (A zero-gradient outflow is no control: in subcritical flow the
    reach would keep whatever depth the transient left.) The steady discharge is checked on the exact boundary
    volumes."""
    cell, w = 5.0, 3
    nx = 400
    hn = swetest.normal_depth(q, n, slope)
    c = swetest.channel(f"normal_{q}_{n}", nx, w, cell, slope=slope, h=1.3 * hn, manning=n,
                        bc={"west": ["inflow", q * w * cell], "east": ["depth", hn]})
    r = run(c, time=20000.0, record_every=1000.0)
    assert_mass(r, tol=1e-5)
    # the first cells feel the inflow side (first-order boundary cell, observed +-1.5 % on the first 3 cells)
    h = r.h[5: nx // 3, 1]
    assert np.abs(h - hn).max() <= 3e-4 * hn, f"h_n = {hn:.5f}, h in [{h.min():.5f}, {h.max():.5f}]"
    # steady: the discharge leaving through the east side over the last interval equals the inflow
    b = swepost.exact_balance(r.rec)
    east = [p["east"] for p in b["parts"]]
    q_out = -(east[-1] - east[-2]) / (b["t"][-1] - b["t"][-2])
    assert abs(q_out - q * w * cell) <= 1e-4 * q * w * cell, f"outflow {q_out:.5f} m3/s for {q * w * cell} m3/s"
