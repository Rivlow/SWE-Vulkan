// Shared solver state: parameters, buffers and indexing.
// Every resource has an explicit binding so all entry points of solver.spv share one descriptor set layout.
// Mirror of src/solver/SolverParams.h - keep both in sync.
module params;

public static const uint FLAG_ZONES    = 1;   // the `cond` buffer holds zone indices (sources, sinks, fixed levels)
public static const uint FLAG_INFLOW   = 2;
public static const uint FLAG_FRICTION = 4;
public static const uint FLAG_EXACT_INFLOW = 8;   // inflow sides: mass flux = imposed unit discharge (not the HLLC flux of a ghost state)
public static const uint FLAG_GROUNDWATER  = 16;  // aquifer under the surface (soil / gw buffers, see groundwater.slang)
public static const uint FLAG_GW_LATERAL   = 32;  //   with lateral (Boussinesq) flow: some cell has kh > 0
public static const uint FLAG_GW_WOLFGPU   = 64;  //   wolfgpu convention for the vertical exchange (h_gw compared with zb + h)
public static const uint FLAG_SOURCES      = 128; // infiltration / rain chronologies on zone maps (src_zones, chrono)
public static const uint FLAG_FACE_BC      = 256; // weak conditions on cell faces (face_bcs, face_list)
public static const uint FLAG_LEGACY_LIMITER = 512; // Venkatakrishnan fed the neighbour increment (Python reference)
public static const uint FLAG_NO_CARRY       = 1024; // depth = the Shu-Osher combination itself (Python reference)
public static const uint FLAG_BRIDGE         = 2048; // bridge decks on some cells (bridge buffer, see bridge.slang)

// per-cell conditions (zones), applied at the end of every step
public static const uint ZONE_NONE   = 0;
public static const uint ZONE_SOURCE = 1;   // value = discharge (m3/s) spread over the zone's cells
public static const uint ZONE_SINK   = 2;   // water removed (outlet)
public static const uint ZONE_LEVEL  = 3;   // value = fixed free-surface elevation (relative to z_ref)
public static const uint MAX_ZONES   = 16;

public struct Zone
{
    public uint  kind;
    public uint  ncells;
    public float value;
    public float _pad;
};

// set_water modes (push constants: coef[0..3] = i0, j0, i1, j1 inclusive, coef[4] = value, mode)
public static const uint WATER_DEPTH = 0;   // h = value
public static const uint WATER_LEVEL = 1;   // h = max(value - zb, 0)
public static const uint WATER_ADD   = 2;   // h += value
public static const uint WATER_CLEAR = 3;   // h = 0

public static const uint BC_WALL   = 0;
public static const uint BC_INFLOW = 1;
public static const uint BC_LEVEL  = 2;
public static const uint BC_FREE   = 3;   // transmissive outflow: ghost = cell (zero gradient)
public static const uint BC_DEPTH  = 4;   // imposed water depth

// Weak conditions on cell faces: a group of faces (cell * 4 + side, side 0 west, 1 east, 2 south, 3 north) that
// behaves like a domain border for the owning cell; the cell across an interior face sees a wall.
public static const uint MAX_FACE_BC = 16;
public static const uint FACE_LIST_GROUP_SHIFT = 28;
public struct FaceBc
{
    public uint  kind;              // BC_*
    public uint  offset;            // first entry in face_list (after the count)
    public uint  count;
    public uint  _pad;
    public float value;             // m3/s (inflow), free-surface level or depth
    public float zmin;              // lowest bed of the owning cells (dry-start inflow)
    public float _p0;
    public float _p1;
};

public struct SolverParams          // std140, 128 bytes
{
    public uint  nx;
    public uint  ny;
    public uint  n_cells;
    public uint  flags;
    public float dx;
    public float dy;
    public float g;
    public float cfl_dx;            // cfl * min(dx, dy), folded in double on the host like Taichi does
    public float h_eps;
    public float h_dry;
    public float h_dry4;            // h_dry^4
    public float eps2;              // Venkatakrishnan epsilon^2
    public float g_n2;              // g * manning_n^2
    public float dt_max;            // cap on the time step (dry domains: s_max -> 0)
    public float bc_fill;           // dry-start inflow: cells within bc_fill (m) above the side's lowest point take the flow
    public float gw_dt_coef;        // groundwater stability: dt <= gw_dt_coef / max(kh h_gw / Sy)
    public uint4  bc_kind;          // west, east, south, north: BC_WALL / BC_INFLOW / BC_LEVEL / BC_FREE
    public float4 bc_value;         // inflow discharge (m3/s) or free-surface level
    public float4 bc_zmin;          // lowest bed elevation of each side
    public float  bridge_slot;      // Preissmann slot of the pressurised bridge cells, relative width (0, 1]
    public float  _pad_b0;
    public float  _pad_b1;
    public float  _pad_b2;
};

// euler_stage flags (StagePC.flags)
public static const uint STAGE_FIRST = 1;   // first stage: saves Q^n into Q_old (multi-stage schemes)
public static const uint STAGE_LAST  = 2;   // last stage: friction, zones, and s_max of the new state for the next step

public struct StagePC               // push constants, 48 bytes
{
    public uint   flags;            // euler_stage: STAGE_*
    public uint   pass;             // drain_alpha: Jacobi pass k; euler_stage: number of passes done (0 = limiter off)
    public uint   mode;             // set_water: WATER_*
    public uint   _pad0;
    public float4 coef[2];          // euler_stage: (a, b) of Q <- a Q^n + b E(Q), then the weight of this stage's
                                    // fluxes in the full step (product of the b of this and the later stages);
                                    // set_water: i0 j0 i1 j1 value
};

// scalars[] slots (uint, floats stored with asuint)
public static const uint S_SMAX_BITS = 0;   // atomic max accumulator: filled by the last stage of a step (or
                                            // reduce_smax after a host change of Q), consumed by finalize_dt
public static const uint S_SMAX      = 1;
public static const uint S_DT        = 2;
public static const uint S_TIME      = 3;   // t, f32 sum (effects)
public static const uint S_NSTEPS    = 4;
public static const uint S_TIME_LO   = 5;   // t exactly, 64-bit fixed point: t = S_TIME_HI 2^-8 + S_TIME_LO 2^-40 (s)
public static const uint S_TIME_HI   = 6;
public static const uint S_GW_D_BITS = 7;   // atomic max of the aquifer diffusivity kh h_gw / Sy (m2/s), like S_SMAX_BITS
public static const uint S_GW_D      = 8;   // value used by the last finalize_dt
public static const uint S_GW_VOLUME = 9;   // diagnostics (diag_final): water in the aquifer, sum Sy h_gw dx dy
public static const uint S_GW_INFIL  = 10;  //   volume infiltrated since t = 0 (surface -> aquifer)
public static const uint S_GW_EXFIL  = 11;  //   volume exfiltrated since t = 0 (aquifer -> surface)
public static const uint S_GW_RATE   = 12;  //   net exchange of the last step (m3/s, > 0 = infiltration)
public static const uint S_SRC_NET   = 13;  // chronologies (diag_final): net volume added to the surface since t = 0
public static const uint S_SRC_RAIN  = 14;  //   rain volume since t = 0
public static const uint S_VOLUME    = 16;  // diagnostics (written by diag_final)
public static const uint S_WET_CELLS = 17;
public static const uint S_HMAX      = 18;
public static const uint S_UMAX      = 19;
public static const uint S_LIMITED   = 20;  // cells with alpha < 1 on the last stage of the step (reset by finalize_dt)
public static const uint S_CLAMP_N   = 21;  // cells whose h had to be clamped to 0 since t = 0 (round-off residuals)
public static const uint S_CLAMP_MAX = 22;  // largest clamped |h| since t = 0 (float bits)
public static const uint S_PROBES    = 32;  // 8 probes x (h, u, v, eta)
public static const uint MAX_PROBES  = 8;
public static const uint S_GW_PROBES = 64;  // groundwater cases: 8 probes x (water table elevation, exchange rate m/s)
// Exact water balance of the open boundaries since t = 0: volume that entered (> 0) or left (< 0) through each
// side (0..3: west, east, south, north) and each group of weak face conditions (4 + g), then the volume each zone
// added or removed. Signed 64-bit fixed point in two words (hi, lo), units of 2^-32 m3 (see track_volume).
public static const uint S_BND_CUM   = 80;  // (4 + MAX_FACE_BC) x 2 words
public static const uint S_ZONE_CUM  = 120; // MAX_ZONES x 2 words

[[vk::binding(0, 0)]] public ConstantBuffer<SolverParams> P;
[[vk::binding(1, 0)]] public RWStructuredBuffer<uint>     scalars;
[[vk::binding(2, 0)]] public StructuredBuffer<float>      zb;
[[vk::binding(3, 0)]] public RWStructuredBuffer<float4>   Q;       // (h, hu, hv, zb): zb copied in w, one load per cell
[[vk::binding(4, 0)]] public RWStructuredBuffer<float4>   Q_old;
[[vk::binding(5, 0)]] public RWStructuredBuffer<float>    alpha_a; // drain limiter: outflow factor per cell, Jacobi ping-pong
[[vk::binding(10, 0)]] public RWStructuredBuffer<float>   alpha_b;
// Face fluxes (h, hu, hv, source): w of F_R / G_U carries the centred bed-slope source Sx / Sy of the cell
// right of / above the face (each cell reads it from its west / south face), 0 in F_L / G_D.
[[vk::binding(6, 0)]] public RWStructuredBuffer<float4>   F_L;     // x-faces (nx+1, ny): flux seen by the left cell
[[vk::binding(7, 0)]] public RWStructuredBuffer<float4>   F_R;     //                     flux seen by the right cell
[[vk::binding(8, 0)]] public RWStructuredBuffer<float4>   G_D;     // y-faces (nx, ny+1): flux seen by the lower cell
[[vk::binding(9, 0)]] public RWStructuredBuffer<float4>   G_U;     //                     flux seen by the upper cell
[[vk::binding(11, 0)]] public StructuredBuffer<uint>      cond;    // zone index + 1 per cell, 0 = none
// Own buffer (not a scalars slot): bc_fluxes runs unsynchronised next to finalize_dt.
[[vk::binding(12, 0)]] public RWStructuredBuffer<float>   inflow;  // sum of h^(5/3) * width per side [0..3], then per face group [4 + g]
[[vk::binding(13, 0)]] public StructuredBuffer<uint>      probe_cells;   // [0] = count, then cell indices
[[vk::binding(14, 0)]] public RWStructuredBuffer<float4>  diag;          // per-workgroup partial diagnostics
[[vk::binding(15, 0)]] public StructuredBuffer<Zone>      zones;         // MAX_ZONES entries
// Drain limiter: mass fluxes of the 4 faces of each cell (west, east, south, north), gathered by pass 0 so that
// the later passes read 16 bytes per cell instead of four 16-byte face fluxes
[[vk::binding(16, 0)]] public RWStructuredBuffer<float4>  mass_flux;
// Active tiles (16 x 16 cells, see solver.slang): the flux, drain and Euler kernels only run on tiles with
// water nearby. Tile t = ti * tiles_y() + tj covers cells i in [16 ti, 16 ti + 15], j in [16 tj, 16 tj + 15].
[[vk::binding(17, 0)]] public RWStructuredBuffer<uint>    tile_wet;      // step count when the tile last held water
[[vk::binding(18, 0)]] public RWStructuredBuffer<uint>    tile_list;     // active tiles of this step (see solver.slang)
[[vk::binding(19, 0)]] public RWStructuredBuffer<uint>    tile_args;     // indirect dispatch (count, 1, 1), then 2 counters
[[vk::binding(20, 0)]] public StructuredBuffer<uint>      tile_seed;     // 1 = always active (sources, levels, open sides)
// Groundwater (FLAG_GROUNDWATER; one element each otherwise). Full grid: the aquifer flows under dry tiles too.
[[vk::binding(21, 0)]] public StructuredBuffer<float4>    soil;          // (kv, kh, Sy, z_sub) per cell
[[vk::binding(22, 0)]] public RWStructuredBuffer<float4>  gw;            // (h_gw, exchange rate m/s (> 0 = infiltration), cumulated infiltration m, cumulated exfiltration m)
// (outflow factor, lateral change of h_gw over the step (float bits), h_gw in 64-bit fixed point: units of 2^-8 m,
// units of 2^-40 m), see groundwater.slang fixed_add
[[vk::binding(23, 0)]] public RWStructuredBuffer<uint4>   gw_aux;
// per-workgroup partial diagnostics of the vertical terms, 2 per group: (aquifer volume, infiltrated,
// exfiltrated, exchange rate), (net volume added by the chronologies, rain volume, 0, 0)
[[vk::binding(24, 0)]] public RWStructuredBuffer<float4>  diag_gw;
// Infiltration / rain chronologies (FLAG_SOURCES; one element each otherwise), as wolfgpu's add_infiltration /
// add_rain: zone maps and tables of discharges over time, already divided by the zone areas (m/s).
[[vk::binding(25, 0)]] public StructuredBuffer<uint>      src_zones;     // infiltration zone | rain zone << 16 (1-based, 0 = none)
// header (lines, zones, lerp, offset) of the infiltration table then of the rain table, then the tables:
// rows of (start time, rate of zone 1, ..., rate of zone n)
[[vk::binding(26, 0)]] public StructuredBuffer<float>     chrono;
[[vk::binding(27, 0)]] public RWStructuredBuffer<float>   src_rates;     // rates of this step: infiltration zones, then rain zones (m/s)
// per cell: net depth added by the chronologies, rain depth, each in signed 64-bit fixed point (groundwater.slang)
[[vk::binding(28, 0)]] public RWStructuredBuffer<uint4>   src_cum;
// Profile along a line of cells (read back each frame like the probes): [0] = count, then cell indices
public static const uint MAX_PROFILE = 2048;
[[vk::binding(29, 0)]] public StructuredBuffer<uint>      profile_cells;
[[vk::binding(30, 0)]] public RWStructuredBuffer<float4>  profile_out;   // (zb, h, water table H, substratum z_sub)
// Weak face conditions (FLAG_FACE_BC; one element each otherwise): MAX_FACE_BC groups, then the face list
// ([0] = total count, then group << FACE_LIST_GROUP_SHIFT | cell * 4 + side)
[[vk::binding(31, 0)]] public StructuredBuffer<FaceBc>    face_bcs;
[[vk::binding(32, 0)]] public StructuredBuffer<uint>      face_list;
// Snapshot for the recorder (pack_record32 / pack_record16; one element until recording is on): per cell three
// words (h, hu, hv as floats) or two (h in mm as uint16 | f16 hu << 16, f16 hv), then one word per tile
// (1 = some water in the tile)
[[vk::binding(33, 0)]] public RWStructuredBuffer<uint>    record;
// Weak face conditions seen from the cells (FLAG_FACE_BC; one element otherwise): per cell, 5 bits per side
// (west, east, south, north) holding the group + 1 of that face, 0 = none. Used to track the boundary volumes.
[[vk::binding(34, 0)]] public StructuredBuffer<uint>      face_cell;
// Exact depth updates (euler_stage): x = this step's weighted sum of the stages' dt L(h) (started from the carry),
// y = carry, the part of the last update that the float32 depth could not hold (TwoSum residual), added back on the
// next step. Without it, increments below half an ulp of h were lost: a slow evolution drifted (1e-4 of the volume
// over 30 000 steps of a backwater curve settling).
[[vk::binding(35, 0)]] public RWStructuredBuffer<float2>  mass_carry;
// Bridges (FLAG_BRIDGE; one element otherwise): per cell (soffit, top of the deck) elevations relative to z_ref,
// >= NO_BRIDGE where there is none
[[vk::binding(36, 0)]] public StructuredBuffer<float2>    bridge_lid;
public static const float NO_BRIDGE = 1e29f;
public bool bridges_on() { return (P.flags & FLAG_BRIDGE) != 0; }
public bool bridge_at(uint c) { return bridges_on() && bridge_lid[c].x < NO_BRIDGE; }

[[vk::push_constant]] public ConstantBuffer<StagePC> pc;

// Row-major (nx, ny) like numpy C order: j varies fastest.
public uint idx(uint i, uint j)  { return i * P.ny + j; }
public uint fidx(uint i, uint j) { return i * P.ny + j; }          // x-face (nx+1, ny)
public uint gidx(uint i, uint j) { return i * (P.ny + 1) + j; }    // y-face (nx, ny+1)

public float stage_coef(uint m) { return pc.coef[m >> 2][m & 3]; }

public static const uint TILE = 16;   // = GRID_THREADS: one workgroup per tile
public uint tiles_x() { return (P.nx + TILE - 1) / TILE; }
public uint tiles_y() { return (P.ny + TILE - 1) / TILE; }

// Jacobi pass k writes alpha buffer k & 1 and reads the other one.
public float read_alpha(uint which, uint c) { return (which & 1) == 0 ? alpha_a[c] : alpha_b[c]; }
public void  write_alpha(uint which, uint c, float a)
{
    if ((which & 1) == 0)
        alpha_a[c] = a;
    else
        alpha_b[c] = a;
}
