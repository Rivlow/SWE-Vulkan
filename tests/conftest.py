"""pytest fixtures of the hydraulic tests: the executable, a work folder per test, and `run`, which writes a case,
runs `swe --headless` on it and hands back the dump (and the recording).

    python -m pytest tests -q                       # release build (build/release/bin/swe.exe)
    python -m pytest tests -q --preset debug
    python -m pytest tests -q -m "not slow"
    python -m pytest tests -q --wolfgpu C:/Users/lucas/Dev/wolfgpu/Repository   # comparisons with wolfgpu

Outputs go to out/tests/<test name>/ (assets of the generated case, dump, recording).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import swepost  # noqa: E402
import swerec  # noqa: E402
import swetest  # noqa: E402
import plots  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--preset", default=os.environ.get("SWE_PRESET", "release"), help="build preset (build/<preset>/bin/swe.exe)")
    default_repo = Path(r"C:\Users\lucas\Dev\wolfgpu\Repository")
    parser.addoption("--exe", default=os.environ.get("SWE_EXE"), help="swe executable (default: build/<preset>/bin/swe.exe)")
    parser.addoption("--no-plots", action="store_true", default=False, help="do not write the PNG plots of the tests")
    parser.addoption("--wolfgpu", default=os.environ.get("WOLFGPU_REPO", str(default_repo) if default_repo.exists() else None),
                     help="wolfgpu repository for the comparison tests (its saved reference results)")
    parser.addoption("--wolfgpu-python", default=os.environ.get("WOLFGPU_PYTHON", r"C:\Users\lucas\miniconda3\envs\wolf_env\python.exe"),
                     help="python of wolfgpu's environment")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long runs (deselect with -m 'not slow')")


@pytest.fixture(scope="session")
def swe_exe(request):
    if request.config.getoption("--exe"):
        return Path(request.config.getoption("--exe"))
    preset = request.config.getoption("--preset")
    exe = ROOT / "build" / preset / "bin" / "swe.exe"
    if not exe.exists():
        exe = exe.with_suffix("")
    if not exe.exists():
        pytest.skip(f"swe executable not found in build/{preset}/bin (build first)")
    return exe


@pytest.fixture(scope="session")
def wolfgpu(request):
    repo = request.config.getoption("--wolfgpu")
    if not repo or not Path(repo).exists():
        pytest.skip("wolfgpu repository not given (--wolfgpu <repo>)")
    return Path(repo), Path(request.config.getoption("--wolfgpu-python"))


@pytest.fixture
def workdir(request):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
    d = ROOT / "out" / "tests" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


@pytest.fixture
def plot(request, workdir):
    """Plotter writing PNGs to <workdir>/plots (tests/plots.py)."""
    return plots.Plotter(workdir / "plots", enabled=not request.config.getoption("--no-plots"))


class Run:
    """Result of a headless run: Q (nx, ny, 3), zb, meta.json, the recording (when asked) and the case."""

    def __init__(self, dump, case=None, record=None, plot=None):
        self.plot = plot or plots.Plotter(Path(dump).parent / "plots", enabled=False)
        self.dump = Path(dump)
        self.Q = np.load(self.dump / "Q.npy")
        self.zb = np.load(self.dump / "zb.npy")
        self.meta = json.loads((self.dump / "meta.json").read_text(encoding="utf-8"))
        self.case = case
        self.rec = swerec.Recording(record) if record else None
        self.gw = np.load(self.dump / "gw.npy") if (self.dump / "gw.npy").exists() else None
        self.sources = np.load(self.dump / "sources.npy") if (self.dump / "sources.npy").exists() else None

    @property
    def h(self):
        return self.Q[..., 0].astype(np.float64)

    @property
    def hu(self):
        return self.Q[..., 1].astype(np.float64)

    @property
    def hv(self):
        return self.Q[..., 2].astype(np.float64)

    @property
    def eta(self):
        return self.h + self.zb.astype(np.float64)

    @property
    def t(self):
        return float(self.meta["t"])

    @property
    def cell(self):
        return float(self.case.cell) if self.case is not None else None

    def area(self):
        return float(self.meta["dx"]) * float(self.meta["dy"])

    def volume(self, stored=None):
        h = self.h
        if stored is not None:
            h = stored(h)
        return float(h.sum() * self.area())

    def stored(self):
        """Physical depth function of the case's bridges (identity without bridges)."""
        if self.case is None or not self.case.bridges:
            return None
        roof, deck = self.case.bridge_arrays()
        zb = self.zb.astype(np.float64)
        slot = self.case.bridge_slot
        return lambda h: swepost.stored_depth(h, zb, roof, deck, slot)


@pytest.fixture
def run(swe_exe, workdir, plot):
    """run(case, steps=..., time=..., scheme=None, drain=None, record_every=None, record_f16=False, all_tiles=False,
    extra=[], name=None) -> Run. `case` is a swetest.Case (written to <workdir>/assets) or the name of a case of
    assets/. The log of swe goes to <workdir>/<name>.log."""

    def _run(case, steps=None, time=None, scheme=None, drain=None, record_every=None, record_f16=False, all_tiles=False,
             extra=(), name=None, assets=None):
        if isinstance(case, swetest.Case):
            assets = workdir / "assets"
            case.write(assets)
            case_name = case.name
        else:
            case_name = case
            assets = assets or (ROOT / "assets")
        tag = name or case_name
        dump = workdir / f"dump_{tag}"
        cmd = [str(swe_exe), "--headless", "--no-validation", "--assets", str(assets), "--case", case_name, "--dump", str(dump)]
        if steps is not None:
            cmd += ["--steps", str(int(steps))]
        if time is not None:
            cmd += ["--time", repr(float(time))]
        if scheme:
            cmd += ["--scheme", scheme]
        if drain is not None:
            cmd += ["--drain", str(int(drain))]
        if all_tiles:
            cmd += ["--all-tiles"]
        record = None
        if record_every is not None:
            record = workdir / f"{tag}.swerec"
            cmd += ["--record", str(record), "--record-every", repr(float(record_every))]
            if record_f16:
                cmd += ["--record-f16"]
        cmd += [str(x) for x in extra]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (workdir / f"{tag}.log").write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr, encoding="utf-8")
        assert r.returncode == 0, f"swe failed ({r.returncode}):\n{' '.join(cmd)}\n{r.stdout}{r.stderr}"
        return Run(dump, case if isinstance(case, swetest.Case) else None, record, plot)

    return _run
