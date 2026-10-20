"""PNG plots of the hydraulic tests: what each test measures, drawn next to its tolerance.

Every test writes its figures to out/tests/<test>/plots/*.png (fixture `plot`, tests/conftest.py). A plot never
makes a test fail: any error while drawing is swallowed (and `--no-plots` skips them all).

    plot.lines(name, x, [("computed", y), ("exact", y_ref, "ref")], title=..., note="L1 = 3e-4 (limit 1e-3)")
    plot.fields(name, [("h (m)", h), ("hu (m2/s)", hu)], extent=...)
    plot.error(name, x, err, tol=1e-3)           # |computed - reference| along x on a log axis with its limit
    plot.convergence(name, dx, errors, tol=...)  # error against the cell size, with the slope 1 and 2
    plot.mass(run, ...)                          # closure of the mass balance (called by swetest.assert_mass)
"""
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:           # pragma: no cover - plots are optional
    plt = None

STYLES = {
    "num": dict(color="#1f77b4", lw=1.6, marker="o", ms=2.5),      # computed by swe
    "ref": dict(color="#d62728", lw=1.4, ls="--"),                  # analytical / reference
    "wolf": dict(color="#2ca02c", lw=1.4, ls="-."),                 # wolfgpu
    "aux": dict(color="#7f7f7f", lw=1.0),
}


def _style(kind, i):
    if kind in STYLES:
        return STYLES[kind]
    return dict(lw=1.4, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10])


class Plotter:
    def __init__(self, folder, enabled=True):
        self.folder = Path(folder)
        self.enabled = enabled and plt is not None

    # -- plumbing -------------------------------------------------------------------------------------------------
    def _save(self, fig, name):
        self.folder.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.folder / f"{name}.png", dpi=110)
        plt.close(fig)

    @staticmethod
    def _note(ax, note):
        if note:
            ax.text(0.01, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb", alpha=0.9))

    def safe(f):                                  # noqa: N805 - decorator of the methods below
        def wrapper(self, *a, **k):
            if not self.enabled:
                return None
            try:
                return f(self, *a, **k)
            except Exception as e:                # never fail a test for a figure
                print(f"[plots] {f.__name__}: {type(e).__name__}: {e}")
                plt.close("all")
                return None
        return wrapper

    # -- generic plots --------------------------------------------------------------------------------------------
    @safe
    def lines(self, name, x, curves, title="", xlabel="x (m)", ylabel="", note="", vlines=(), hlines=(), logy=False,
              xlim=None):
        """curves: [(label, y[, kind])] with kind in num / ref / wolf / aux (or None). `x` may be an array or a list
        of arrays, one per curve. vlines / hlines: [(position, label)]."""
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, c in enumerate(curves):
            label, y = c[0], c[1]
            kind = c[2] if len(c) > 2 else ("ref" if "exact" in label or "ref" in label.lower() else "num")
            xi = x[i] if isinstance(x, (list, tuple)) else x
            st = dict(_style(kind, i))
            if len(y) > 150:
                st.pop("marker", None)
            ax.plot(xi, y, label=label, **st)
        for pos, lab in vlines:
            ax.axvline(pos, color="k", lw=0.8, ls=":")
            ax.text(pos, 1.0, f" {lab}", transform=ax.get_xaxis_transform(), va="top", fontsize=7)
        for pos, lab in hlines:
            ax.axhline(pos, color="k", lw=0.8, ls=":")
            ax.text(0.99, pos, lab, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7)
        if logy:
            ax.set_yscale("log")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        self._note(ax, note)
        self._save(fig, name)

    @safe
    def error(self, name, x, err, tol=None, title="", xlabel="x (m)", ylabel="|computed - reference|", note=""):
        """Pointwise error on a log axis, with the tolerance of the test as a line."""
        err = np.abs(np.asarray(err, dtype=np.float64))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.semilogy(x, np.maximum(err, 1e-18), **STYLES["num"])
        if tol is not None:
            ax.axhline(tol, color="#d62728", ls="--", lw=1.2, label=f"limit {tol:.1e}")
            ax.legend(fontsize=8)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3, which="both")
        self._note(ax, note or f"max = {err.max():.2e}")
        self._save(fig, name)

    @safe
    def convergence(self, name, dx, errors, tol=None, title="", ylabel="relative L1 error", note=""):
        """Error against cell size on log-log axes with the slopes 1 and 2 as guides."""
        dx, errors = np.asarray(dx, float), np.asarray(errors, float)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.loglog(dx, errors, label="swe", **STYLES["num"])
        for p, ls in ((1, ":"), (2, "-.")):
            ax.loglog(dx, errors[0] * (dx / dx[0]) ** p, color="#7f7f7f", ls=ls, lw=1, label=f"slope {p}")
        if tol is not None:
            ax.axhline(tol, color="#d62728", ls="--", lw=1.2, label=f"limit {tol:.1e}")
        ax.set(title=title, xlabel="cell size dx (m)", ylabel=ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        self._note(ax, note)
        self._save(fig, name)

    @safe
    def fields(self, name, panels, extent=None, title="", cmap="viridis", symmetric=False, note=""):
        """panels: [(label, 2-D array (nx, ny))]; drawn with x to the right (the arrays are indexed [i, j])."""
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)
        for ax, (label, a) in zip(axes[0], panels):
            a = np.asarray(a, dtype=np.float64)
            kw = {}
            if symmetric:
                m = max(np.nanmax(np.abs(a)), 1e-30)
                kw = dict(vmin=-m, vmax=m, cmap="RdBu_r")
            im = ax.imshow(a.T, origin="lower", extent=extent, aspect="equal", cmap=kw.pop("cmap", cmap), **kw)
            ax.set_title(label, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if title:
            fig.suptitle(title, fontsize=10)
        if note:
            fig.text(0.01, 0.01, note, fontsize=8)
        self._save(fig, name)

    @safe
    def bars(self, name, items, title="", ylabel="", note=""):
        """items: [(label, value, limit or None)] on a log axis: measured value (blue) against its limit (red)."""
        fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(items)), 3.6))
        xs = np.arange(len(items))
        vals = [max(float(v), 1e-18) for _, v, _ in items]
        ax.bar(xs, vals, color="#1f77b4", width=0.55)
        for k, (_, _, lim) in enumerate(items):
            if lim is not None:
                ax.plot([k - 0.4, k + 0.4], [lim, lim], color="#d62728", lw=2)
        ax.set_yscale("log")
        ax.set_xticks(xs, [i[0] for i in items], rotation=20, ha="right", fontsize=8)
        ax.set(title=title, ylabel=ylabel)
        ax.grid(alpha=0.3, axis="y", which="both")
        self._note(ax, note or "blue: measured, red: limit of the test")
        self._save(fig, name)

    # -- mass -----------------------------------------------------------------------------------------------------
    @safe
    def mass(self, run, err, tol, volume0, V, added, stored=None, series_tol=None):
        """Mass closure: V(t) against V0 + what entered (exact boundary volumes), and the relative residual of every
        record against its limit. Without recording: the final closure error against the limit."""
        import swepost
        tag = run.dump.name.removeprefix("dump_")
        b = swepost.exact_balance(run.rec, stored) if run.rec is not None else None
        if b is None:
            self.bars(f"mass_closure_{tag}", [("final closure", max(err, 1e-18), tol)],
                      title="Mass closure |V - V0 - added| / scale", ylabel="relative error")
            return
        scale = max(abs(volume0), abs(V), abs(added), 1e-30)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        a1.plot(b["t"], b["V"], label="V(t) in the domain", **STYLES["num"])
        a1.plot(b["t"], b["V"][0] + b["added"] - b["added"][0], label="V0 + volume entered through the boundaries", **STYLES["ref"])
        a1.set(ylabel="volume (m3)", title="Mass balance: the two curves must superpose")
        a1.legend(fontsize=8)
        a1.grid(alpha=0.3)
        res = np.abs(b["residual"]) / scale
        a2.semilogy(b["t"], np.maximum(res, 1e-18), **STYLES["num"])
        a2.axhline(series_tol or tol, color="#d62728", ls="--", label=f"limit {(series_tol or tol):.1e}")
        a2.set(xlabel="t (s)", ylabel="|V - V0 - added| / scale")
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3, which="both")
        self._note(a2, f"max {res.max():.2e}")
        self._save(fig, f"mass_closure_{tag}")
