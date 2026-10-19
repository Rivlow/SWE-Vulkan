"""Post-processing of the .swerec recordings of SWE-Vulkan (tools/swerec.py reads them).

Library (every function returns float64 numpy arrays; `rec` is a swerec.Recording):

    import swepost
    rec = swepost.Recording("records/canal_20261019_120000.swerec")
    t, V = swepost.volume_series(rec)                       # water volume (m3) of every record
    t, Q = swepost.section_discharge(rec, "x", 25, 0, 9)   # discharge (m3/s) through the x-faces i = 25, j in 0..9
    res = swepost.hydrograph_balance(rec, inflows=[("x", 0, 0, 9)], outflows=[("x", 50, 0, 9)])
    g = swepost.gauge(rec, 10, 5, zb=zb)                    # t, h, eta, u, v, Fr at a cell
    env = swepost.envelopes(rec)                            # h max, |u| max, eta max, h |u| max, arrival, duration
    rows = swepost.mass_report(rec, inflows, outflows)      # per record: V, dV, integrated Q, residual

Mass balance is the first thing the tests check: `hydrograph_balance` returns, for every record k,
    V(t_k) - V(t_0) - integral_0^k (Q_in - Q_out) dt   (trapezoidal rule on the recorded discharges)
which is zero up to the time discretisation of Q(t) when the solver conserves mass (a closed domain gives
V(t_k) - V(t_0) exactly).

Bridges: the solver's h is the piezometric depth; the stored (physical) depth is `stored_depth(h, zb, roof, deck,
slot)` (see README, "Ponts"): pass `stored=lambda h: swepost.stored_depth(h, zb, roof, deck, slot)` to the volume
functions to get the physical volume.

Command line:
    python tools/swepost.py file.swerec volume [--zb zb.npy]
    python tools/swepost.py file.swerec section x 25 0 9
    python tools/swepost.py file.swerec gauge 10 5 [--zb zb.npy] [--csv out.csv]
    python tools/swepost.py file.swerec envelope --out out_dir [--zb zb.npy] [--geotiff case.json]
    python tools/swepost.py file.swerec diff other.swerec
    python tools/swepost.py file.swerec animate out.mp4 [--field h|speed|eta] [--zb zb.npy] [--fps 20]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swerec import Recording  # noqa: E402

G = 9.81


# ---------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------

def load_zb(rec, zb=None, dump=None, assets=None):
    """Bed elevation (nx, ny) for the recording: an array, a .npy file, a --dump folder (zb.npy) or the assets
    folder of the case (imported cases only: data/<name>.npz). Returns None when nothing is available."""
    if zb is not None:
        a = np.load(zb) if isinstance(zb, (str, Path)) else np.asarray(zb)
        return a.astype(np.float64)
    if dump is not None:
        return np.load(Path(dump) / "zb.npy").astype(np.float64)
    if assets is not None:
        case = json.loads((Path(assets) / "cases" / f"{rec.case}.json").read_text(encoding="utf-8"))
        if case.get("type") == "imported":
            return np.load(Path(assets) / case["data"])["zb"].astype(np.float64)
    return None


def load_bridges(assets, case_name):
    """(roof, deck, slot) of a case: per-cell elevations relative to z_ref (NaN = no bridge), or None."""
    path = Path(assets) / "cases" / f"{case_name}.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("type") != "imported":
        return None
    npz = np.load(Path(assets) / case["data"])
    if "bridge_roof" not in npz:
        return None
    return npz["bridge_roof"].astype(np.float64), npz["bridge_deck"].astype(np.float64), float(case.get("bridge_slot", 0.05))


def stored_depth(h, zb, roof, deck, slot=0.05):
    """Physical (stored) depth of the bridge model from the piezometric depth h: h below the soffit; the clearance
    c = roof - zb plus a Preissmann slot of relative width `slot` while pressurised; the clearance, the slot and the
    layer above the deck when overtopping. Cells without a bridge (roof NaN) store h."""
    h = np.asarray(h, dtype=np.float64)
    roof = np.asarray(roof, dtype=np.float64)
    bridge = np.isfinite(roof)
    # cells without a bridge: a clearance above any depth (finite, no inf arithmetic), then h is returned anyway
    c = np.where(bridge, np.maximum(np.where(bridge, roof, 0.0) - zb, 0.0), np.max(h, initial=0.0) + 1.0)
    d = np.where(bridge, np.maximum(np.where(bridge, np.asarray(deck, dtype=np.float64) - roof, 0.0), 0.0), 0.0)
    s = np.where(h <= c, h, np.where(h <= c + d, c + slot * (h - c), c + slot * d + (h - c - d)))
    return np.where(bridge, s, h)


def effective_depth(h, zb, roof, deck):
    """Flowing depth of the bridge model: min(h, c) + max(h - c - d, 0)."""
    h = np.asarray(h, dtype=np.float64)
    roof = np.asarray(roof, dtype=np.float64)
    bridge = np.isfinite(roof)
    c = np.where(bridge, np.maximum(np.where(bridge, roof, 0.0) - zb, 0.0), np.max(h, initial=0.0) + 1.0)
    d = np.where(bridge, np.maximum(np.where(bridge, np.asarray(deck, dtype=np.float64) - roof, 0.0), 0.0), 0.0)
    return np.where(bridge, np.minimum(h, c) + np.maximum(h - c - d, 0.0), h)


def velocity(h, q, h_dry=1e-3):
    """Velocity q / h, 0 on dry cells (same desingularisation threshold as the solver's rendering)."""
    h = np.asarray(h, dtype=np.float64)
    return np.where(h > h_dry, np.asarray(q, dtype=np.float64) / np.maximum(h, 1e-12), 0.0)


# ---------------------------------------------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------------------------------------------

def volume_series(rec, stored=None):
    """t (s) and water volume V (m3) of every record; `stored(h) -> s` converts to the physical depth (bridges)."""
    area = rec.dx * rec.dy
    t = np.asarray(rec.times, dtype=np.float64)
    V = np.empty(len(rec))
    for k in range(len(rec)):
        h, _, _ = rec.read(k)
        h = h.astype(np.float64)
        if stored is not None:
            h = stored(h)
        V[k] = h.sum() * area
    return t, V


def section_discharge(rec, axis, index, lo, hi):
    """Discharge (m3/s, positive along +axis) through a line of faces of every record.
    axis "x": the x-faces i = index (between cells index-1 and index), rows j in [lo, hi];
    axis "y": the y-faces j = index, columns i in [lo, hi]. Interior faces take the mean of the two cells' unit
    discharges; index 0 / nx (or ny) is a boundary and takes the boundary cell's value."""
    n = rec.nx if axis == "x" else rec.ny
    width = rec.dy if axis == "x" else rec.dx
    t = np.asarray(rec.times, dtype=np.float64)
    Q = np.empty(len(rec))
    for k in range(len(rec)):
        _, qx, qy = rec.read(k)
        q = (qx if axis == "x" else qy).astype(np.float64)
        if axis == "y":
            q = q.T        # (ny, nx): the face index is now the first axis
        if index <= 0:
            line = q[0, lo:hi + 1]
        elif index >= n:
            line = q[n - 1, lo:hi + 1]
        else:
            line = 0.5 * (q[index - 1, lo:hi + 1] + q[index, lo:hi + 1])
        Q[k] = line.sum() * width
    return t, Q


def _sections_total(rec, sections, sign):
    total = np.zeros(len(rec))
    for axis, index, lo, hi in sections:
        _, Q = section_discharge(rec, axis, index, lo, hi)
        total += sign(axis, index) * Q
    return total


def hydrograph_balance(rec, inflows=(), outflows=(), stored=None):
    """Residual of the mass balance per record: V(t_k) - V(t_0) - integral (Q_in - Q_out) dt.
    `inflows` / `outflows`: lists of sections (axis, index, lo, hi); the discharge of an inflow section is counted
    positive when it enters the domain through it (a section at index 0 enters along +axis, one at nx along -axis;
    an interior inflow section is counted along +axis), an outflow section positive when water leaves.
    Returns dict(t, V, Qin, Qout, residual, residual_rel)."""
    t, V = volume_series(rec, stored)
    n = (rec.nx, rec.ny)

    def entering(axis, index):
        return -1.0 if index >= n[0 if axis == "x" else 1] else 1.0

    def leaving(axis, index):
        return -1.0 if index <= 0 else 1.0

    Qin = _sections_total(rec, inflows, entering)
    Qout = _sections_total(rec, outflows, leaving)
    net = Qin - Qout
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (net[1:] + net[:-1]) * np.diff(t))])
    residual = V - V[0] - integral
    scale = max(V[0], float(np.abs(integral).max()) if len(integral) else 0.0, 1e-30)
    return {"t": t, "V": V, "Qin": Qin, "Qout": Qout, "residual": residual, "residual_rel": residual / scale}


def exact_balance(rec, stored=None):
    """Mass balance from the volumes the solver tracks through every open side, face group and zone (records of
    version 2): residual_k = V(t_k) - V(t_0) - (added(t_k) - added(t_0)), round-off only when mass is conserved.
    Returns dict(t, V, added, residual, residual_rel, parts) or None for version 1 files."""
    if not getattr(rec, "scalar_words", 0):
        return None
    t, V = volume_series(rec, stored)
    added = np.array([rec.added_volume(k) for k in range(len(rec))])
    residual = V - V[0] - (added - added[0])
    scale = max(float(np.abs(V).max()), float(np.abs(added - added[0]).max()), 1e-30)
    parts = [rec.boundary_volumes(k) for k in range(len(rec))]
    return {"t": t, "V": V, "added": added, "residual": residual, "residual_rel": residual / scale, "parts": parts}


def mass_report(rec, inflows=(), outflows=(), stored=None):
    """Rows (t, V, dV, integral, residual, residual_rel) per record: the exact balance of the solver's boundary
    volumes when the file has them (version 2) and no section is given, else the balance of the sections."""
    if not inflows and not outflows:
        e = exact_balance(rec, stored)
        if e is not None:
            return [{"t": float(e["t"][k]), "V": float(e["V"][k]), "dV": float(e["V"][k] - e["V"][0]),
                     "integral": float(e["added"][k] - e["added"][0]), "residual": float(e["residual"][k]),
                     "residual_rel": float(e["residual_rel"][k])} for k in range(len(rec))]
    b = hydrograph_balance(rec, inflows, outflows, stored)
    integral = b["V"] - b["V"][0] - b["residual"]
    rows = [{"t": float(b["t"][k]), "V": float(b["V"][k]), "dV": float(b["V"][k] - b["V"][0]),
             "integral": float(integral[k]), "Qin": float(b["Qin"][k]), "Qout": float(b["Qout"][k]),
             "residual": float(b["residual"][k]), "residual_rel": float(b["residual_rel"][k])} for k in range(len(rec))]
    return rows


def gauge(rec, i, j, zb=None, h_dry=1e-3):
    """Time series at cell (i, j): dict(t, h, eta, u, v, speed, Fr); eta needs zb (else NaN)."""
    t = np.asarray(rec.times, dtype=np.float64)
    h = np.empty(len(rec))
    u = np.empty(len(rec))
    v = np.empty(len(rec))
    for k in range(len(rec)):
        hk, qx, qy = rec.read(k)
        h[k] = hk[i, j]
        u[k] = velocity(hk[i, j], qx[i, j], h_dry)
        v[k] = velocity(hk[i, j], qy[i, j], h_dry)
    speed = np.hypot(u, v)
    Fr = np.where(h > h_dry, speed / np.sqrt(G * np.maximum(h, 1e-12)), 0.0)
    eta = h + (zb[i, j] if zb is not None else np.nan)
    return {"t": t, "h": h, "eta": eta, "u": u, "v": v, "speed": speed, "Fr": Fr}


def profile(rec, k, axis, index, zb=None, h_dry=1e-3):
    """Profile of record k along x at row j = index (axis "x") or along y at column i = index: dict(s, zb, h, eta,
    u_normal) with s the distance along the line (cell centres)."""
    h, qx, qy = rec.read(k)
    if axis == "x":
        hl, ql = h[:, index], qx[:, index]
        zl = zb[:, index] if zb is not None else np.zeros(rec.nx)
        s = (np.arange(rec.nx) + 0.5) * rec.dx
    else:
        hl, ql = h[index, :], qy[index, :]
        zl = zb[index, :] if zb is not None else np.zeros(rec.ny)
        s = (np.arange(rec.ny) + 0.5) * rec.dy
    hl = hl.astype(np.float64)
    return {"s": s, "zb": zl.astype(np.float64), "h": hl, "eta": hl + zl, "u": velocity(hl, ql, h_dry)}


def envelopes(rec, h_wet=1e-3, zb=None):
    """Maxima over the recording and timing of the flood: dict(h_max, speed_max, eta_max, hazard_max (h |u|),
    arrival (s, first record with h > h_wet, NaN if never), duration (s, time with h > h_wet))."""
    shape = (rec.nx, rec.ny)
    h_max = np.zeros(shape)
    speed_max = np.zeros(shape)
    hazard_max = np.zeros(shape)
    arrival = np.full(shape, np.nan)
    duration = np.zeros(shape)
    t = np.asarray(rec.times, dtype=np.float64)
    for k in range(len(rec)):
        h, qx, qy = rec.read(k)
        h = h.astype(np.float64)
        speed = np.hypot(velocity(h, qx), velocity(h, qy))
        wet = h > h_wet
        h_max = np.maximum(h_max, h)
        speed_max = np.maximum(speed_max, speed)
        hazard_max = np.maximum(hazard_max, h * speed)
        arrival = np.where(wet & np.isnan(arrival), t[k], arrival)
        if k + 1 < len(rec):
            duration += wet * (t[k + 1] - t[k])
    out = {"h_max": h_max, "speed_max": speed_max, "hazard_max": hazard_max, "arrival": arrival, "duration": duration}
    if zb is not None:
        out["eta_max"] = np.where(h_max > h_wet, h_max + zb, np.nan)
    return out


def diff(rec_a, rec_b, h_wet=1e-3):
    """Differences of two recordings of the same grid: dict(dh_max (a - b), flooded_a, flooded_b, newly_flooded,
    area_a, area_b (m2))."""
    if (rec_a.nx, rec_a.ny) != (rec_b.nx, rec_b.ny):
        raise ValueError("different grids")
    ea, eb = envelopes(rec_a, h_wet), envelopes(rec_b, h_wet)
    fa, fb = ea["h_max"] > h_wet, eb["h_max"] > h_wet
    area = rec_a.dx * rec_a.dy
    return {"dh_max": ea["h_max"] - eb["h_max"], "flooded_a": fa, "flooded_b": fb, "newly_flooded": fa & ~fb,
            "area_a": float(fa.sum() * area), "area_b": float(fb.sum() * area)}


# ---------------------------------------------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------------------------------------------

def export_npy(fields, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, a in fields.items():
        np.save(out_dir / f"{name}.npy", np.asarray(a))


def export_csv(series, path, keys=None):
    """Writes the columns of a dict of equal-length arrays (gauge, volume_series...)."""
    keys = keys or [k for k, v in series.items() if np.ndim(v) == 1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*(np.asarray(series[k]) for k in keys)):
            w.writerow([f"{x:.9g}" for x in row])


def export_geotiff(field, path, case_json, nodata=np.nan):
    """Writes an (nx, ny) field as a GeoTIFF with the origin, cell and crs of an imported case (needs GDAL)."""
    from osgeo import gdal, osr
    case = json.loads(Path(case_json).read_text(encoding="utf-8"))
    x0, y0 = case["origin"]
    cell = float(case["cell"])
    epsg = int(case.get("crs", "EPSG:31370").split(":")[1])
    a = np.asarray(field, dtype=np.float32)
    nx, ny = a.shape
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), nx, ny, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
    ds.SetGeoTransform((x0, cell, 0.0, y0 + ny * cell, 0.0, -cell))        # north up: row 0 = top
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(a.T[::-1])        # (ny, nx), rows from the north
    if nodata is not None and np.isfinite(nodata):
        band.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None


def animate(rec, path, field="h", zb=None, fps=20, vmax=None, ortho=None, dpi=100):
    """Animation (mp4 via ffmpeg, else gif) of h, speed or eta over the recording (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    extent = (0, rec.nx * rec.dx, 0, rec.ny * rec.dy)
    fig, ax = plt.subplots(figsize=(8, 8 * rec.ny / max(rec.nx, 1)), dpi=dpi)
    if ortho is not None:
        ax.imshow(plt.imread(ortho), extent=extent, origin="upper")

    def frame_field(k):
        h, qx, qy = rec.read(k)
        h = h.astype(np.float64)
        if field == "speed":
            a = np.hypot(velocity(h, qx), velocity(h, qy))
        elif field == "eta":
            a = np.where(h > 1e-3, h + (zb if zb is not None else 0.0), np.nan)
        else:
            a = np.where(h > 1e-3, h, np.nan)
        return a.T

    first = frame_field(0)
    if vmax is None:
        vmax = max(float(np.nanmax(frame_field(k))) if np.isfinite(frame_field(k)).any() else 0.0 for k in range(len(rec)))
        vmax = vmax or 1.0
    im = ax.imshow(first, origin="lower", extent=extent, vmin=0 if field != "eta" else None, vmax=vmax, cmap="Blues" if field != "speed" else "viridis",
                   alpha=0.85 if ortho is not None else 1.0)
    fig.colorbar(im, ax=ax, label={"h": "h (m)", "speed": "|u| (m/s)", "eta": "eta (m)"}[field])
    title = ax.set_title(f"{rec.case}: t = {rec.times[0]:.2f} s")

    def update(k):
        im.set_data(frame_field(k))
        title.set_text(f"{rec.case}: t = {rec.times[k]:.2f} s")
        return im, title

    anim = animation.FuncAnimation(fig, update, frames=len(rec), blit=False)
    path = Path(path)
    if path.suffix.lower() == ".gif":
        anim.save(str(path), writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(str(path), writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--zb", help="zb.npy of the case (--dump folder) for eta")
    ap.add_argument("--dump", help="--dump folder holding zb.npy")
    ap.add_argument("--assets", help="assets folder of the case (imported cases: zb and bridges from the npz)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("volume", help="volume of every record")
    p = sub.add_parser("section", help="discharge through a line of faces")
    p.add_argument("axis", choices=["x", "y"])
    p.add_argument("index", type=int)
    p.add_argument("lo", type=int)
    p.add_argument("hi", type=int)
    p.add_argument("--csv")
    p = sub.add_parser("gauge", help="time series at a cell")
    p.add_argument("i", type=int)
    p.add_argument("j", type=int)
    p.add_argument("--csv")
    p = sub.add_parser("envelope", help="h max, |u| max, hazard, arrival, duration")
    p.add_argument("--out", required=True)
    p.add_argument("--geotiff", help="case JSON (imported case): also write GeoTIFFs")
    p.add_argument("--wet", type=float, default=1e-3)
    p = sub.add_parser("diff", help="compare with another recording")
    p.add_argument("other")
    p = sub.add_parser("animate", help="animation of a field")
    p.add_argument("out")
    p.add_argument("--field", choices=["h", "speed", "eta"], default="h")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--ortho", help="orthophoto PNG of the case as background")
    args = ap.parse_args(argv)

    rec = Recording(args.file)
    zb = load_zb(rec, args.zb, args.dump, args.assets)
    stored = None
    if args.assets:
        br = load_bridges(args.assets, rec.case)
        if br is not None and zb is not None:
            roof, deck, slot = br
            stored = lambda h: stored_depth(h, zb, roof, deck, slot)        # noqa: E731

    if args.cmd == "volume":
        print(f"{rec.path.name}: {len(rec)} records, {rec.nx} x {rec.ny} cells of {rec.dx:g} m")
        e = exact_balance(rec, stored)
        if e is None:
            t, V = volume_series(rec, stored)
            for k in range(len(rec)):
                print(f"  {k:4d}  t = {t[k]:10.3f} s  V = {V[k]:14.6f} m3  dV = {V[k] - V[0]:+.6e} m3  ({(V[k] - V[0]) / max(V[0], 1e-30):+.3e})")
        else:
            print("  (balance = V - V0 - volume added by the open boundaries and the zones; 0 up to round-off)")
            for k in range(len(rec)):
                parts = "  ".join(f"{n} {v:+.4f}" for n, v in e["parts"][k].items() if v != 0.0)
                print(f"  {k:4d}  t = {e['t'][k]:10.3f} s  V = {e['V'][k]:14.6f} m3  added {e['added'][k]:+14.6f}  "
                      f"balance {e['residual'][k]:+.3e} ({e['residual_rel'][k]:+.2e})  {parts}")
    elif args.cmd == "section":
        t, Q = section_discharge(rec, args.axis, args.index, args.lo, args.hi)
        for k in range(len(rec)):
            print(f"  {k:4d}  t = {t[k]:10.3f} s  Q = {Q[k]:12.6f} m3/s")
        if args.csv:
            export_csv({"t": t, "Q": Q}, args.csv)
    elif args.cmd == "gauge":
        g = gauge(rec, args.i, args.j, zb)
        for k in range(len(rec)):
            print(f"  {k:4d}  t = {g['t'][k]:10.3f} s  h = {g['h'][k]:.4f} m  eta = {g['eta'][k]:.4f}  |u| = {g['speed'][k]:.4f} m/s  Fr = {g['Fr'][k]:.3f}")
        if args.csv:
            export_csv(g, args.csv)
    elif args.cmd == "envelope":
        env = envelopes(rec, args.wet, zb)
        export_npy(env, args.out)
        if args.geotiff:
            for name, a in env.items():
                export_geotiff(a, Path(args.out) / f"{name}.tif", args.geotiff)
        area = rec.dx * rec.dy
        print(f"flooded area {float((env['h_max'] > args.wet).sum() * area):.1f} m2, h max {env['h_max'].max():.3f} m, "
              f"|u| max {env['speed_max'].max():.3f} m/s, hazard max {env['hazard_max'].max():.3f} m2/s -> {args.out}")
    elif args.cmd == "diff":
        d = diff(rec, Recording(args.other))
        print(f"flooded area: {d['area_a']:.1f} m2 vs {d['area_b']:.1f} m2, newly flooded {float(d['newly_flooded'].sum() * rec.dx * rec.dy):.1f} m2, "
              f"dh max in [{d['dh_max'].min():+.3f}, {d['dh_max'].max():+.3f}] m")
    elif args.cmd == "animate":
        animate(rec, args.out, args.field, zb, args.fps, ortho=args.ortho)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
