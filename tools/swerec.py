"""Reader of the .swerec recordings written by SWE-Vulkan (Simulation stage > Enregistrement, or --record).

    import swerec
    rec = swerec.Recording("records/zone_importee_20261018_170000.swerec")
    print(rec.header)                # nx, ny, dx, dy, z_ref, case name, number of records
    print(rec.times)                 # simulated time of every record
    h, qx, qy = rec.read(10)         # dense float32 arrays (nx, ny), zeros where dry
    for k, (t, h, qx, qy) in enumerate(rec):
        ...

    python tools/swerec.py file.swerec            # summary and volume of every record
    python tools/swerec.py file.swerec --plot 5   # h of record 5 (matplotlib)

Layout (little endian): a 256-byte header, the records, then an index of (offset, t, step, tiles) entries.
A record holds the 16 x 16 tiles with water only: a header (t, step, tiles), from version 2 the solver scalars
(header field scalar_words uint32: exact volumes through every boundary and zone, see boundary_volumes), the
tile ids (ti * tilesY + tj),
then per tile 256 cells (di major, dj minor), each (h, hu, hv) as three float32 (encoding 2, exact) or
(uint16 h in mm, float16 hu, float16 hv) (encoding 1), zero outside the grid.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HEADER = np.dtype([
    ("magic", "S8"), ("version", "<u4"),
    ("nx", "<u4"), ("ny", "<u4"), ("tile", "<u4"), ("tilesX", "<u4"), ("tilesY", "<u4"),
    ("encoding", "<u4"), ("pad0", "<u4"),
    ("dx", "<f8"), ("dy", "<f8"), ("z_ref", "<f8"),
    ("index_offset", "<u8"), ("count", "<u4"), ("scalar_words", "<u4"),
    ("case", "S64"), ("reserved", "V112"),
])
RECORD = np.dtype([("t", "<f8"), ("step", "<u4"), ("tiles", "<u4")])
INDEX = np.dtype([("offset", "<u8"), ("t", "<f8"), ("step", "<u4"), ("tiles", "<u4")])
CELL16 = np.dtype([("h", "<u2"), ("hu", "<f2"), ("hv", "<f2")])
CELL32 = np.dtype([("h", "<f4"), ("hu", "<f4"), ("hv", "<f4")])

# slots of the solver scalars carried by each record (version 2, src/solver/SolverParams.h ScalarSlot)
S_TIME_LO, S_TIME_HI, S_NSTEPS = 5, 6, 4
S_CLAMP_N = 21
S_BND_CUM, S_ZONE_CUM = 80, 120
MAX_FACE_BC, MAX_ZONES = 16, 16
SIDES = ("west", "east", "south", "north")


def fixed_volume(words, slot):
    """Signed 64-bit fixed-point volume (hi, lo) of the scalars, units of 2^-32 m3."""
    v = (int(words[slot]) << 32) | int(words[slot + 1])
    if v >= 1 << 63:
        v -= 1 << 64
    return v / 4294967296.0


class Recording:
    def __init__(self, path):
        self.path = Path(path)
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        hdr = np.frombuffer(self.mm[:HEADER.itemsize], dtype=HEADER)[0]
        if bytes(hdr["magic"]).rstrip(b"\0") != b"SWEREC1":        # numpy strips the trailing NUL of S8 fields
            raise ValueError(f"{path}: not a .swerec file")
        self.nx, self.ny, self.tile = int(hdr["nx"]), int(hdr["ny"]), int(hdr["tile"])
        self.tiles_x, self.tiles_y = int(hdr["tilesX"]), int(hdr["tilesY"])
        self.dx, self.dy, self.z_ref = float(hdr["dx"]), float(hdr["dy"]), float(hdr["z_ref"])
        self.case = hdr["case"].split(b"\0", 1)[0].decode()
        self.encoding = int(hdr["encoding"])
        if self.encoding not in (1, 2):
            raise ValueError(f"{path}: unknown encoding {self.encoding}")
        self.cell = CELL16 if self.encoding == 1 else CELL32
        self.h_scale = 1e-3 if self.encoding == 1 else 1.0
        self.version = int(hdr["version"])
        self.scalar_words = int(hdr["scalar_words"]) if self.version >= 2 else 0
        count, off = int(hdr["count"]), int(hdr["index_offset"])
        if count == 0 and off == 0:
            raise ValueError(f"{path}: recording still open or not closed properly (no index)")
        self.index = np.frombuffer(self.mm[off:off + count * INDEX.itemsize], dtype=INDEX)
        self.header = {"case": self.case, "nx": self.nx, "ny": self.ny, "dx": self.dx, "dy": self.dy,
                       "z_ref": self.z_ref, "records": count, "encoding": "float16" if self.encoding == 1 else "float32"}

    @property
    def times(self):
        return self.index["t"]

    def __len__(self):
        return len(self.index)

    def read(self, k):
        """Dense (nx, ny) float32 arrays h (m), qx, qy (m2/s) of record k."""
        e = self.index[k]
        off = int(e["offset"])
        rh = np.frombuffer(self.mm[off:off + RECORD.itemsize], dtype=RECORD)[0]
        n = int(rh["tiles"])
        off += RECORD.itemsize + 4 * self.scalar_words
        ids = np.frombuffer(self.mm[off:off + 4 * n], dtype="<u4")
        off += 4 * n
        cells = np.frombuffer(self.mm[off:off + n * self.tile * self.tile * self.cell.itemsize], dtype=self.cell)
        cells = cells.reshape(n, self.tile, self.tile)
        T = self.tile
        h = np.zeros((self.tiles_x * T, self.tiles_y * T), np.float32)
        qx = np.zeros_like(h)
        qy = np.zeros_like(h)
        ti, tj = ids // self.tiles_y, ids % self.tiles_y
        for a, (i, j) in enumerate(zip(ti, tj)):
            sl = (slice(i * T, (i + 1) * T), slice(j * T, (j + 1) * T))
            h[sl] = cells["h"][a].astype(np.float32) * self.h_scale
            qx[sl] = cells["hu"][a].astype(np.float32)
            qy[sl] = cells["hv"][a].astype(np.float32)
        return h[:self.nx, :self.ny], qx[:self.nx, :self.ny], qy[:self.nx, :self.ny]

    def __iter__(self):
        for k in range(len(self)):
            yield (float(self.index[k]["t"]),) + self.read(k)

    def scalars(self, k):
        """Raw solver scalars (uint32) of record k (version 2), None for version 1 files."""
        if not self.scalar_words:
            return None
        off = int(self.index[k]["offset"]) + RECORD.itemsize
        return np.frombuffer(self.mm[off:off + 4 * self.scalar_words], dtype="<u4")

    def boundary_volumes(self, k):
        """Exact volumes (m3, > 0 = entered) since t = 0 at record k: dict side name / "group <g>" / "zone <z>"
        -> volume (version 2 files; None otherwise). The sum is what the open boundaries and zones added."""
        s = self.scalars(k)
        if s is None:
            return None
        out = {}
        for g in range(4 + MAX_FACE_BC):
            v = fixed_volume(s, S_BND_CUM + 2 * g)
            if v != 0.0 or g < 4:
                out[SIDES[g] if g < 4 else f"group {g - 4}"] = v
        for z in range(MAX_ZONES):
            v = fixed_volume(s, S_ZONE_CUM + 2 * z)
            if v != 0.0:
                out[f"zone {z}"] = v
        return out

    def added_volume(self, k):
        """Total volume added since t = 0 by the open boundaries and the zones at record k (m3)."""
        b = self.boundary_volumes(k)
        return None if b is None else float(sum(b.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--plot", type=int, default=None, help="show h of this record (matplotlib)")
    args = ap.parse_args()
    rec = Recording(args.file)
    print(f"{rec.path.name}: case {rec.case}, {rec.nx} x {rec.ny} cells of {rec.dx:g} m, {len(rec)} records, "
          f"{rec.header['encoding']}")
    area = rec.dx * rec.dy
    for k, e in enumerate(rec.index):
        h, qx, qy = rec.read(k)
        print(f"  {k:4d}  t = {e['t']:10.3f} s  step {e['step']:8d}  tiles {e['tiles']:6d}  "
              f"volume {h.sum() * area:12.3f} m3  h max {h.max():.3f} m  |q| max {np.hypot(qx, qy).max():.3f} m2/s")
    if args.plot is not None:
        import matplotlib.pyplot as plt
        h, _, _ = rec.read(args.plot)
        plt.imshow(h.T, origin="lower", extent=(0, rec.nx * rec.dx, 0, rec.ny * rec.dy))
        plt.colorbar(label="h (m)")
        plt.title(f"{rec.case}: t = {rec.times[args.plot]:.2f} s")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
