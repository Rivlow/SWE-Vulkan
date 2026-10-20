"""Recordings (.swerec) written by --headless runs, and the post-processing that the other tests rely on."""
import numpy as np

import swepost
import swerec


def test_record_times_and_final_state(run, plot):
    r = run("dam_break", time=1.0, record_every=0.1)
    t = np.asarray(r.rec.times)
    plot.lines("record_times", np.arange(len(t)), [("record time", t, "num"), ("due: 0.1 k", 0.1 * np.arange(len(t)), "ref")], xlabel="record k", ylabel="t (s)",
               title="Recording times: each record is the first state past a multiple of the interval", note="lateness (t_k - 0.1 k) is below one step")
    assert t[0] == 0.0
    assert len(r.rec) >= 10
    assert np.all(np.diff(t) > 0)
    # each record is the first state past a multiple of the interval (within one step)
    k = np.arange(1, len(t))
    assert np.all(t[1:] >= 0.1 * k - 1e-9)
    # the last record is the final state when it was due: same bits as the dump
    if abs(t[-1] - r.t) < 1e-9:
        h, qx, qy = r.rec.read(len(r.rec) - 1)
        assert np.array_equal(h, r.Q[..., 0]) and np.array_equal(qx, r.Q[..., 1]) and np.array_equal(qy, r.Q[..., 2])


def test_record_f16_accuracy(run, plot):
    a = run("dam_break", time=0.5, record_every=0.25, name="f32")
    b = run("dam_break", time=0.5, record_every=0.25, record_f16=True, name="f16")
    assert len(a.rec) == len(b.rec)
    ha_, _, _ = a.rec.read(len(a.rec) - 1)
    hb_, _, _ = b.rec.read(len(b.rec) - 1)
    plot.fields("f16", [("h float32 (m)", ha_), ("h float16 - h float32 (m)", hb_ - ha_)], title="float16 recording: depth error (limit 5e-4 m)",
                note=f"max |dh| = {np.abs(ha_ - hb_).max():.1e}")
    for k in range(len(a.rec)):
        ha, qxa, _ = a.rec.read(k)
        hb, qxb, _ = b.rec.read(k)
        assert np.abs(ha - hb).max() <= 0.5e-3 + 1e-7
        assert np.abs(qxa - qxb).max() <= 1e-3 * max(np.abs(qxa).max(), 1e-6) + 1e-4


def test_record_dir(swe_exe, workdir):
    import subprocess
    out = workdir / "records"
    r = subprocess.run([str(swe_exe), "--headless", "--no-validation", "--case", "dam_break", "--steps", "50",
                        "--record-dir", str(out), "--record-every", "0.05"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    files = list(out.glob("dam_break_*.swerec"))
    assert len(files) == 1
    assert len(swerec.Recording(files[0])) >= 2


def test_section_discharge_matches_inflow(run, plot):
    """Post-processing check: the discharge through a section is the imposed discharge once the flow is steady
    (uniform flow at normal depth started at equilibrium). swepost estimates it from the cell-centred unit
    discharges, not from the face mass fluxes the solver exchanges: the friction step and the first-order boundary
    cell make them differ by a few 1e-3 (observed 1.3e-3 next to the inflow side, 5e-4 inside), hence 5e-3. Mass
    itself is checked exactly on volumes (test_mass_conservation)."""
    import swetest
    nx, ny, cell, q, n, s = 60, 3, 1.0, 1.0, 0.03, 1e-3
    hn = swetest.normal_depth(q, n, s)
    c = swetest.channel("section_q", nx, ny, cell, slope=s, h=hn, q=q, manning=n,
                        bc={"west": ["inflow", q * ny * cell], "east": ["depth", hn]})
    r = run(c, time=200.0, record_every=50.0)
    for index in (1, nx // 2, nx - 1):
        tq, Q = swepost.section_discharge(r.rec, "x", index, 0, ny - 1)
        plot.lines(f"section_{index}", tq, [(f"Q through face {index}", Q, "num"), ("imposed inflow", np.full_like(tq, q * ny * cell), "ref")], xlabel="t (s)",
                   ylabel="Q (m3/s)", title=f"Discharge through a section estimated from the recording (face {index})", note="limit: 5e-3 relative")
        assert abs(Q[-1] - q * ny * cell) <= 5e-3 * q * ny * cell, f"face {index}: Q = {Q[-1]}"
    h = r.h[:, 1]
    assert np.abs(h - hn).max() <= 5e-3 * hn, f"normal depth {hn:.5f}, computed {h.min():.5f}..{h.max():.5f}"
