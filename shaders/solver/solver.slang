// Port of solver/time_integration.py + compute_flux / apply_BC: one compute entry point per parallel loop.
// The host (src/solver/Solver.cpp) chains them per step, then per Runge-Kutta stage, with barriers:
//
//   build_tiles                                   active tile list + indirect dispatch arguments
//   [finalize_dt] flux_x flux_y bc_fluxes         (flux kernels also write the bed-slope source)
//   drain_alpha x K passes                        (drain limiter, Jacobi ping-pong on alpha_a / alpha_b)
//   euler_stage                                   Q <- a Q^n + b (Q + dt L(Q)), fix_dry [, friction, zones]
//   [gw_alpha gw_lateral] vertical_step           rain / infiltration chronologies and groundwater, once per step,
//                                                 full grid (see groundwater.slang)
//
// s_max of the next step is reduced by the last euler_stage from the state it writes (no extra pass over Q);
// reduce_smax only runs after the host changed Q (load, reset, water edits). The first stage saves Q^n.
//
// Active tiles. flux_x, flux_y, drain_alpha and euler_stage run one workgroup per active 16 x 16 tile
// (indirect dispatch), with results identical to the bit to a full-grid run:
//  - a tile is active when it or one of its 8 neighbours held water (any non-zero component) at the start of
//    the step, or when it is seeded (source / level zone cells, sides with an inflow or level condition);
//  - an Euler stage only moves water to the adjacent cells (a face between two dry cells has exactly zero
//    flux: constant reconstruction, zero depths), so water travels at most one cell per stage, 3 per step,
//    and never reaches an inactive tile within the step: its cells stay exactly dry and unchanged, and the
//    faces between it and an active tile carry exactly zero flux - read as zero instead of from memory, since
//    the inactive tile does not write the faces it owns (east / north faces of the active tile's edge cells);
//  - "held water" comes from tile_wet, stamped with the step count by the last euler_stage (or reduce_smax
//    after a host change of Q) for every tile it leaves wet: no clearing pass.
//
// Time integration is in Shu-Osher form: every SSP scheme is a convex combination of Euler steps, so once an
// Euler step keeps h >= 0 the whole step does, exactly, in floating point too (a, b >= 0).
//
// Drain limiter (keeps h >= 0 without creating mass): each cell gets one factor alpha in [0, 1] that scales
// all its outgoing face fluxes (both sides of a face see the same scaled flux -> conservation). alpha solves
// the monotone fixed point  alpha_i = min(1, (h_i + dt I_i(alpha)) / (dt O_i))  where O_i is the raw outflow and
// I_i the inflow accepted by the upstream cells, by K Jacobi passes started from alpha = 0: every pass is already
// positivity-safe, passes only increase alpha (more flux admitted). The same formula is wolfgpu's alpha correction,
// iterated upward instead of downward.
//
// Threads: j = tid.x, i = tid.y so neighbouring threads touch neighbouring addresses (j is fastest).
import params;
import reconstruction;
import flux;
import boundary_condition;
import groundwater;

#define GRID_THREADS [numthreads(16, 16, 1)]

// ---------------------------------------------------------------------------------------------
// Active tiles
// ---------------------------------------------------------------------------------------------

// List entries: (ti << 17) | (tj << 2) | EAST_ACTIVE | NORTH_ACTIVE (state of the tiles (ti + 1, tj), (ti, tj + 1))
static const uint EAST_ACTIVE  = 1;
static const uint NORTH_ACTIVE = 2;

// Cell of a thread of an active tile
struct TileCell
{
    uint i, j;
    uint tile;         // ti * tiles_y() + tj
    uint neighbours;   // EAST_ACTIVE | NORTH_ACTIVE
};

// Cell of thread `lt` in the active tile number `slot` of this step's list
TileCell active_cell(uint slot, uint2 lt)
{
    uint     packed = tile_list[slot];
    uint     ti = packed >> 17, tj = (packed >> 2) & 0x7fffu;
    TileCell c;
    c.i          = ti * TILE + lt.y;
    c.j          = tj * TILE + lt.x;
    c.tile       = ti * tiles_y() + tj;
    c.neighbours = packed & 3u;
    return c;
}

// Faces owned by the neighbouring tile (east face of its last column, north face of its last row): computed
// only when that tile is active, exactly zero otherwise. The domain's own east / north faces are boundary faces.
bool east_face_live(TileCell c)
{
    return ((c.i + 1) % TILE) != 0 || c.i + 1 >= P.nx || (c.neighbours & EAST_ACTIVE) != 0;
}
bool north_face_live(TileCell c)
{
    return ((c.j + 1) % TILE) != 0 || c.j + 1 >= P.ny || (c.neighbours & NORTH_ACTIVE) != 0;
}

bool is_wet(float3 q)
{
    return any(q != 0.0f);
}

// Stamps the tile of this workgroup as holding water at the end of step `step` (one store per wave)
void stamp_wet(bool wet, uint tile, uint step)
{
    if (WaveActiveAnyTrue(wet) && WaveIsFirstLane())
        tile_wet[tile] = step;
}

// Active list of this step. One thread per tile. tile_args = (count, 1, 1 | running count, groups done):
// the groups append to the running count; the last group to finish publishes it as the dispatch count and
// resets both counters for the next step - no host-side clear, no extra barrier.
static const uint TILE_ARGS_RUNNING = 3;
static const uint TILE_ARGS_DONE    = 4;

// Activity rule: seeded, or water in the tile or one of its 8 neighbours at the end of the previous step
// (`wet` = window of wet flags around (ti, tj), wet[a][b] for tile (ti + a - 1, tj + b - 1))
bool tile_is_active(int ti, int tj, int a, int b, bool wet[4][4])
{
    if (ti < 0 || tj < 0 || ti >= int(tiles_x()) || tj >= int(tiles_y()))
        return false;
    if (tile_seed[uint(ti) * tiles_y() + uint(tj)] != 0)
        return true;
    for (int da = -1; da <= 1; ++da)
        for (int db = -1; db <= 1; ++db)
            if (wet[a + da][b + db])
                return true;
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void build_tiles(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    uint nty = tiles_y(), ntx = tiles_x();
    uint t   = tid.x;
    bool active = false;
    uint entry  = 0;
    if (t < ntx * nty)
    {
        int  ti = int(t / nty), tj = int(t % nty);
        uint step = scalars[S_NSTEPS];        // the previous step's count: finalize_dt has not run yet
        // wet flags of tiles (ti - 1 .. ti + 2, tj - 1 .. tj + 2): this tile and its east / north neighbours
        // are decided from the same data, so the neighbour bits agree with the neighbours' own decision
        bool wet[4][4];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
            {
                int ni = ti + a - 1, nj = tj + b - 1;
                wet[a][b] = ni >= 0 && nj >= 0 && ni < int(ntx) && nj < int(nty) && tile_wet[uint(ni) * nty + uint(nj)] == step;
            }
        active = tile_is_active(ti, tj, 1, 1, wet);
        if (active)
            entry = (uint(ti) << 17) | (uint(tj) << 2) | (tile_is_active(ti + 1, tj, 2, 1, wet) ? EAST_ACTIVE : 0u) |
                    (tile_is_active(ti, tj + 1, 1, 2, wet) ? NORTH_ACTIVE : 0u);
    }
    // compaction: one atomic per wave
    uint n    = WaveActiveCountBits(active);
    uint base = 0;
    if (WaveIsFirstLane() && n > 0)
        InterlockedAdd(tile_args[TILE_ARGS_RUNNING], n, base);
    base = WaveReadLaneFirst(base);
    if (active)
        tile_list[base + WavePrefixCountBits(active)] = entry;

    // every append of this group is done (and visible device-wide) before the group counts itself done
    DeviceMemoryBarrierWithGroupSync();
    if (gi == 0)
    {
        uint groups = (tiles_x() * tiles_y() + 63) / 64;
        uint done;
        InterlockedAdd(tile_args[TILE_ARGS_DONE], 1, done);
        if (done == groups - 1)
        {
            uint total;
            InterlockedExchange(tile_args[TILE_ARGS_RUNNING], 0, total);
            tile_args[0]              = total;
            tile_args[TILE_ARGS_DONE] = 0;
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Time step
// ---------------------------------------------------------------------------------------------

// |u|max + sqrt(g h) of a wet cell (h >= h_dry), 0 for thin films: they do not drive the time step
float cell_wave_speed(float3 q)
{
    float h = q.x;
    if (h < P.h_dry)
        return 0.0f;
    float u = velocity(h, q.y);
    float v = velocity(h, q.z);
    return wave_speed(h, max(abs(u), abs(v)));
}

// Folds a speed into the s_max accumulator. Positive floats compare like their bit patterns, so a uint
// atomic max is exact (no float atomics).
void accumulate_smax(float s)
{
    float m = WaveActiveMax(s);
    if (WaveIsFirstLane())
        InterlockedMax(scalars[S_SMAX_BITS], asuint(m));
}

// Same for the aquifer diffusivity (groundwater stability limit)
void accumulate_gw_diffusivity(float d)
{
    float m = WaveActiveMax(d);
    if (WaveIsFirstLane())
        InterlockedMax(scalars[S_GW_D_BITS], asuint(m));
}

// s_max and wet tiles of the current state, after the host changed Q (the steps themselves do it in
// euler_stage). Full grid: workgroup (gid.y, gid.x) = tile (ti, tj).
[shader("compute")]
GRID_THREADS
void reduce_smax(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    uint  j = tid.x, i = tid.y;
    float s = 0.0f, d = 0.0f;
    bool  wet = false;
    if (i < P.nx && j < P.ny)
    {
        float3 q = Q[idx(i, j)].xyz;
        s   = cell_wave_speed(q);
        wet = is_wet(q);
        if (gw_lateral())
            d = gw_diffusivity(soil[idx(i, j)], gw[idx(i, j)].x);
    }
    accumulate_smax(s);
    if (gw_lateral())
        accumulate_gw_diffusivity(d);
    stamp_wet(wet, gid.y * tiles_y() + gid.x, scalars[S_NSTEPS]);
}

// dt = cfl min(dx, dy) / s_max, t += dt. Resets the accumulator for the next step.
// S_TIME is the plain f32 sum (effects); the clock read by the host is also summed exactly in 64-bit fixed
// point, 2^-40 s resolution, over two uints (S_TIME_HI = units of 2^-8 s, S_TIME_LO = units of 2^-40 s):
// an f32 sum loses hours-long runs (ulp(3600 s) = 2.4e-4 s against dt ~ 0.01 s), and integer additions
// cannot be "simplified" by the compiler the way a compensated float sum can.
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize_dt()
{
    float s_max = max(asfloat(scalars[S_SMAX_BITS]), 1e-6f);
    float dt    = min(P.cfl_dx / s_max, P.dt_max);
    if (gw_lateral())
    {
        // explicit aquifer flow: dt <= Sy / (4 T (1/dx^2 + 1/dy^2)) at every cell
        float d = asfloat(scalars[S_GW_D_BITS]);
        if (d > 0.0f)
            dt = min(dt, P.gw_dt_coef / d);
        scalars[S_GW_D_BITS] = 0;
        scalars[S_GW_D]      = asuint(d);
    }
    if (sources_on())
    {
        // chronology rates of this step, at its start time (wolfgpu: simulation_time before the step)
        float t = float(scalars[S_TIME_HI]) * (1.0f / 256.0f) + float(scalars[S_TIME_LO]) * 9.094947e-13f;   // 2^-40
        chrono_rates(0, t);
        chrono_rates(1, t);
    }
    float ds    = dt * 256.0f;                              // exact (power of two)
    float whole = floor(ds);
    uint  hi    = uint(whole);
    uint  lo    = uint((ds - whole) * 4294967296.0f);       // exact difference, scaled by 2^32: < 2^32
    uint  sum   = scalars[S_TIME_LO] + lo;
    scalars[S_TIME_HI]   = scalars[S_TIME_HI] + hi + (sum < lo ? 1u : 0u);
    scalars[S_TIME_LO]   = sum;
    scalars[S_SMAX_BITS] = 0;
    scalars[S_SMAX]      = asuint(s_max);
    scalars[S_DT]        = asuint(dt);
    scalars[S_TIME]      = asuint(asfloat(scalars[S_TIME]) + dt);
    scalars[S_NSTEPS]    = scalars[S_NSTEPS] + 1;
    scalars[S_LIMITED]   = 0;
}

float step_dt() { return asfloat(scalars[S_DT]); }

// ---------------------------------------------------------------------------------------------
// Spatial operator
// ---------------------------------------------------------------------------------------------

// Centred bed-slope source (Audusse 2004, 2nd order) of a cell along one direction: S = g/2 (h_- + h_+)(zb_- - zb_+)
// from the cell's own reconstructed face values; zero when the reconstruction is constant (boundary cells).
float bed_source(Recon r)
{
    return 0.5f * P.g * (r.m.x + r.p.x) * (r.m.w - r.p.w);
}

// Interior x-faces i in [1, nx-1]: face i lies between cells (i-1, j) and (i, j).
// Also writes the x source of the right cell (i, j) in F_R.w: its reconstruction is already at hand, and
// every cell i >= 1 has exactly one such face; cell 0 is a boundary cell (source 0, written by bc_fluxes).
[shader("compute")]
GRID_THREADS
void flux_x(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID)
{
    TileCell tc = active_cell(gid.x, lt.xy);
    uint     i = tc.i, j = tc.j;
    if (i < 1 || i >= P.nx || j >= P.ny)
        return;
    float g = P.g;

    Recon  r_left  = reconstruct_X(i - 1, j);   // p: east face of the left cell
    Recon  r_right = reconstruct_X(i, j);       // m: west face of the right cell
    float3 q_left  = r_left.p.xyz;
    float3 q_right = r_right.m.xyz;
    float  z_left = r_left.p.w, z_right = r_right.m.w;
    float  z_face = max(z_left, z_right);

    float3 q_left_star  = hydrostatic_reconstruct(q_left, z_left, z_face);
    float3 q_right_star = hydrostatic_reconstruct(q_right, z_right, z_face);

    float3 f = Flux_x(q_left_star, q_right_star);

    uint k = fidx(i, j);
    F_L[k] = float4(f + float3(0.0f, 0.5f * g * (q_left.x * q_left.x - q_left_star.x * q_left_star.x), 0.0f), 0.0f);
    F_R[k] = float4(f + float3(0.0f, 0.5f * g * (q_right.x * q_right.x - q_right_star.x * q_right_star.x), 0.0f),
                    bed_source(r_right));
}

// Interior y-faces j in [1, ny-1]: face j lies between cells (i, j-1) and (i, j). G_U.w = y source of (i, j).
[shader("compute")]
GRID_THREADS
void flux_y(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID)
{
    TileCell tc = active_cell(gid.x, lt.xy);
    uint     i = tc.i, j = tc.j;
    if (i >= P.nx || j < 1 || j >= P.ny)
        return;
    float g = P.g;

    Recon  r_down = reconstruct_Y(i, j - 1);    // p: north face of the lower cell
    Recon  r_up   = reconstruct_Y(i, j);        // m: south face of the upper cell
    float3 q_down = r_down.p.xyz;
    float3 q_up   = r_up.m.xyz;
    float  z_down = r_down.p.w, z_up = r_up.m.w;
    float  z_face = max(z_down, z_up);

    float3 q_down_star = hydrostatic_reconstruct(q_down, z_down, z_face);
    float3 q_up_star   = hydrostatic_reconstruct(q_up, z_up, z_face);

    float3 f = Flux_y(q_down_star, q_up_star);

    uint k = gidx(i, j);
    G_D[k] = float4(f + float3(0.0f, 0.0f, 0.5f * g * (q_down.x * q_down.x - q_down_star.x * q_down_star.x)), 0.0f);
    G_U[k] = float4(f + float3(0.0f, 0.0f, 0.5f * g * (q_up.x * q_up.x - q_up_star.x * q_up_star.x)),
                    bed_source(r_up));
}

// With FLAG_EXACT_INFLOW the mass flux of an inflow side is the imposed unit discharge (positive inwards);
// the momentum fluxes keep the HLLC values of the ghost state.
float3 boundary_flux(uint side, float3 f, float3 q, float z)
{
    if ((P.flags & FLAG_EXACT_INFLOW) != 0 && P.bc_kind[side] == BC_INFLOW)
        f.x = INWARD[side] * inflow_unit_discharge(side, q, z);
    return f;
}

// Cell next to the boundary face k of a side (west/east: row j = k, south/north: column i = k)
uint side_cell(uint side, uint k)
{
    return side == 0 ? idx(0, k) : side == 1 ? idx(P.nx - 1, k) : side == 2 ? idx(k, 0) : idx(k, P.ny - 1);
}

groupshared float g_partial[256];

// Boundary faces, one workgroup per side (west, east, south, north). An inflow side first reduces its weights
// sum(h^(5/3) * width) in a fixed order (deterministic, unlike float atomics), then every thread computes faces.
// Self-contained (reads Q only, writes the boundary faces, which flux_x / flux_y never touch): runs in the same
// phase as the interior fluxes, no barrier of its own.
[shader("compute")]
[numthreads(256, 1, 1)]
void bc_fluxes(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID)
{
    uint side = gid.x;
    uint t    = ltid.x;
    uint n    = side < 2 ? P.ny : P.nx;           // west/east sides run along j
    if (P.bc_kind[side] == BC_INFLOW)             // uniform per workgroup
    {
        float width = side < 2 ? P.dy : P.dx;
        float acc   = 0.0f;
        for (uint k = t; k < n; k += 256)
        {
            uint c = side_cell(side, k);
            acc += inflow_weight(inflow_depth(side, Q[c].x, zb[c])) * width;
        }
        g_partial[t] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128; stride > 0; stride >>= 1)
        {
            if (t < stride)
                g_partial[t] += g_partial[t + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (t == 0)
            inflow[side] = g_partial[0];
        AllMemoryBarrierWithGroupSync();          // inflow[side] visible to the whole group (inflow_norm)
    }
    for (uint k = t; k < n; k += 256)
    {
        uint   c = side_cell(side, k);
        float3 q = Q[c].xyz;
        float  z = zb[c];
        if (side == 0)          // cell (0, j) is right of the face
            F_R[fidx(0, k)] = float4(boundary_flux(0, Flux_x(ghost(0, q, z), q), q, z), 0.0f);
        else if (side == 1)     // cell (nx-1, j) is left of the face
            F_L[fidx(P.nx, k)] = float4(boundary_flux(1, Flux_x(q, ghost(1, q, z)), q, z), 0.0f);
        else if (side == 2)     // cell (i, 0) is above the face
            G_U[gidx(k, 0)] = float4(boundary_flux(2, Flux_y(ghost(2, q, z), q), q, z), 0.0f);
        else                    // cell (i, ny-1) is below the face
            G_D[gidx(k, P.ny)] = float4(boundary_flux(3, Flux_y(q, ghost(3, q, z)), q, z), 0.0f);
    }
}

// ---------------------------------------------------------------------------------------------
// Drain limiter
// ---------------------------------------------------------------------------------------------

// Mass fluxes of the four faces of cell (i, j), per unit face length, positive along +x / +y:
// x = west face, y = east face, z = south face, w = north face.
float4 face_mass_fluxes(TileCell c)
{
    uint i = c.i, j = c.j;
    return float4(F_R[fidx(i, j)].x, east_face_live(c) ? F_L[fidx(i + 1, j)].x : 0.0f,
                  G_U[gidx(i, j)].x, north_face_live(c) ? G_D[gidx(i, j + 1)].x : 0.0f);
}

// Raw outflow rate of the cell (m/s): what leaves through every face if nothing is limited.
float outflow_rate(float4 f)
{
    return (max(f.y, 0.0f) + max(-f.x, 0.0f)) / P.dx + (max(f.w, 0.0f) + max(-f.z, 0.0f)) / P.dy;
}

// Factors of the four donor cells across the faces (1 across the domain boundary: inflow from a
// boundary condition is never limited).
float4 neighbour_alphas(uint which, uint i, uint j)
{
    return float4(i > 0        ? read_alpha(which, idx(i - 1, j)) : 1.0f,
                  i + 1 < P.nx ? read_alpha(which, idx(i + 1, j)) : 1.0f,
                  j > 0        ? read_alpha(which, idx(i, j - 1)) : 1.0f,
                  j + 1 < P.ny ? read_alpha(which, idx(i, j + 1)) : 1.0f);
}

// Inflow rate accepted by the upstream cells (m/s), given their factors. Same expression in drain_alpha and
// euler_stage so that both evaluate it to the same bits.
float inflow_rate(float4 f, float4 a)
{
    return (max(f.x, 0.0f) * a.x + max(-f.y, 0.0f) * a.y) / P.dx
         + (max(f.z, 0.0f) * a.z + max(-f.w, 0.0f) * a.w) / P.dy;
}

// Jacobi pass pc.pass: alpha_i = min(1, (h_i + dt I_i) / (dt O_i)), I_i from the previous pass (0 on pass 0).
// Factors of the cells of inactive tiles are stale but finite, and only ever multiply their exactly-zero faces.
[shader("compute")]
GRID_THREADS
void drain_alpha(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID)
{
    TileCell tc = active_cell(gid.x, lt.xy);
    uint     i = tc.i, j = tc.j;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c  = idx(i, j);
    float  dt = step_dt();
    float4 f;
    if (pc.pass == 0)
    {
        f            = face_mass_fluxes(tc);
        mass_flux[c] = f;
    }
    else
        f = mass_flux[c];
    float  O  = outflow_rate(f);
    float  a  = 1.0f;
    float  h  = Q[c].x;
    // h >= dt O: the cell cannot drain whatever it receives (I >= 0), alpha = 1 exactly as the formula would
    // give in floating point too, without reading the neighbours' factors - the common case by far
    if (O > 0.0f && h < dt * O)
    {
        float I = pc.pass > 0 ? inflow_rate(f, neighbour_alphas(pc.pass + 1, i, j)) : 0.0f;
        a = min(1.0f, (h + dt * I) / (dt * O));
    }
    write_alpha(pc.pass, c, a);
}

// ---------------------------------------------------------------------------------------------
// Euler stage (pointwise)
// ---------------------------------------------------------------------------------------------

float3 fix_dry(float3 q)
{
    float h = q.x;
    if (h < P.h_eps)                    // dry cell: momentum cut
        return float3(max(h, 0.0f), 0.0f, 0.0f);
    if (h < P.h_dry)                    // thin film: momentum from the desingularised velocity
        return float3(h, h * velocity(h, q.y), h * velocity(h, q.z));
    return q;
}

// Manning bed friction, semi-implicit (fractional step): stable for any dt
float3 friction(float3 q, float dt)
{
    float h = q.x;
    if (h > P.h_eps)
    {
        float u  = velocity(h, q.y);
        float v  = velocity(h, q.z);
        float cf = P.g_n2 * sqrt(u * u + v * v) / pow(h, 4.0f / 3.0f);
        q.y /= 1.0f + dt * cf;
        q.z /= 1.0f + dt * cf;
    }
    return q;
}

// One Runge-Kutta stage in Shu-Osher form: Q <- a Q^n + b (Q + dt L(Q)), then fix_dry; on the last stage
// (STAGE_LAST) friction and the per-cell zones (sources, sinks, fixed levels) as well, and the wave speed of
// the new state goes into the s_max accumulator of the next step.
// L(Q) = -(flux balance) / area + source, with every face flux scaled by the factor of its donor cell when
// the drain limiter is on (pc.pass = number of Jacobi passes done).
[shader("compute")]
GRID_THREADS
void euler_stage(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID)
{
    TileCell tc = active_cell(gid.x, lt.xy);
    uint     i = tc.i, j = tc.j;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c  = idx(i, j);
    float  dt = step_dt();
    float4 q4 = Q[c];
    float3 q  = q4.xyz;
    // multi-stage schemes combine with Q^n: saved here, where the cell is read anyway (pointwise, the later
    // stages read it after a barrier)
    if ((pc.flags & (STAGE_FIRST | STAGE_LAST)) == STAGE_FIRST)
        Q_old[c] = q4;
    // the cell is right of its west face (F_R) and left of its east face (F_L); w = bed-slope source
    float4 Fw4 = F_R[fidx(i, j)], Gs4 = G_U[gidx(i, j)];
    float2 sb  = float2(Fw4.w, Gs4.w);
    float3 Fw = Fw4.xyz, Fe = east_face_live(tc) ? F_L[fidx(i + 1, j)].xyz : float3(0.0f);
    float3 Gs = Gs4.xyz, Gn = north_face_live(tc) ? G_D[gidx(i, j + 1)].xyz : float3(0.0f);

    bool  limited = false;
    float h_e     = 0.0f;
    if (pc.pass > 0)
    {
        uint   cur = pc.pass + 1;                   // buffer of the last pass (pass k wrote buffer k & 1)
        float4 f   = float4(Fw.x, Fe.x, Gs.x, Gn.x);
        float  a_c = read_alpha(cur, c);
        float4 a_n = neighbour_alphas(cur, i, j);
        // each face is scaled by the factor of the cell the water comes from
        Fw *= f.x >= 0.0f ? a_n.x : a_c;
        Fe *= f.y >= 0.0f ? a_c : a_n.y;
        Gs *= f.z >= 0.0f ? a_n.z : a_c;
        Gn *= f.w >= 0.0f ? a_c : a_n.w;
        limited = a_c < 1.0f;
        if (limited)
        {
            // alpha was set so that h + dt (I_prev - alpha O) == 0: write that zero exactly instead of the
            // rounded difference; what is left is the inflow admitted since the pass alpha was computed
            // (monotone passes -> I_cur >= I_prev, in floating point too).
            float I_cur  = inflow_rate(f, a_n);
            float I_prev = pc.pass > 1 ? inflow_rate(f, neighbour_alphas(cur + 1, i, j)) : 0.0f;
            h_e = dt * max(I_cur - I_prev, 0.0f);
        }
    }
    float3 L = -(Fe - Fw) / P.dx - (Gn - Gs) / P.dy + float3(0.0f, sb.x / P.dx, sb.y / P.dy);
    if (!limited)
        h_e = q.x + dt * L.x;
    if (h_e < 0.0f)
    {
        // Limiter off: the usual clamp (creates mass). Limiter on: only a round-off residual of an unlimited
        // cell right at the margin can land here; counted so it stays visible.
        InterlockedAdd(scalars[S_CLAMP_N], 1);
        InterlockedMax(scalars[S_CLAMP_MAX], asuint(-h_e));
        h_e = 0.0f;
    }
    bool last = (pc.flags & STAGE_LAST) != 0;
    if (limited && last)
        InterlockedAdd(scalars[S_LIMITED], 1);

    float3 qe = float3(h_e, q.y + dt * L.y, q.z + dt * L.z);
    // Convex combination a Q^n + b qe written as Q^n + b (qe - Q^n): h >= 0 holds exactly in floating point
    // (b <= 1 so b (Q^n - qe) <= Q^n when qe < Q^n), and no mass bias when a + b != 1 in f32 (ssprk3: 1/3 + 2/3,
    // which as a h + b h drifted the volume by 3e-8 per step).
    float3 qn = qe;
    if (stage_coef(0) != 0.0f)
    {
        float3 qo = Q_old[c].xyz;
        qn = qo + stage_coef(1) * (qe - qo);
    }
    qn = fix_dry(qn);
    if (last)
    {
        if ((P.flags & FLAG_FRICTION) != 0)
            qn = friction(qn, dt);
        if ((P.flags & FLAG_ZONES) != 0 && cond[c] != 0)
        {
            Zone zn = zones[min(cond[c] - 1, MAX_ZONES - 1)];
            if (zn.kind == ZONE_SINK)
                qn = float3(0.0f);
            else if (zn.kind == ZONE_SOURCE && zn.ncells > 0)
                qn.x += zn.value * dt / (float(zn.ncells) * P.dx * P.dy);      // volume per step shared by the zone
            else if (zn.kind == ZONE_LEVEL)
                qn = float3(max(zn.value - q4.w, 0.0f), 0.0f, 0.0f);          // reservoir at rest
        }
        accumulate_smax(cell_wave_speed(qn));
        stamp_wet(is_wet(qn), tc.tile, scalars[S_NSTEPS]);
    }
    Q[c] = float4(qn, q4.w);
}

// ---------------------------------------------------------------------------------------------
// Groundwater (groundwater.slang): full grid, once per step after the last Euler stage
// ---------------------------------------------------------------------------------------------

// Outflow factor of each cell: alpha = min(1, Sy h_gw / (dt O)), O = raw lateral outflow rate
[shader("compute")]
GRID_THREADS
void gw_alpha(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c  = idx(i, j);
    float4 sc = soil[c];
    float  hc = gw[c].x;
    float  O  = 0.0f;
    for (uint k = 0; k < 4; ++k)
    {
        uint n;
        if (gw_neighbour(i, j, k, n))
            O += max(lateral_rate(sc, hc, soil[n], gw[n].x, gw_inv_d2(k)), 0.0f);
    }
    float stored = sc.z * hc, dt = step_dt();
    gw_aux[c].x = asuint(O > 0.0f && stored < dt * O ? stored / (dt * O) : 1.0f);
}

// Change of h_gw over the step from the lateral fluxes, each scaled by the factor of its donor cell
[shader("compute")]
GRID_THREADS
void gw_lateral(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c   = idx(i, j);
    float4 sc  = soil[c];
    float  hc  = gw[c].x;
    float  a_c = asfloat(gw_aux[c].x);
    float  net = 0.0f;        // outflow rate (m/s over the cell area)
    for (uint k = 0; k < 4; ++k)
    {
        uint n;
        if (gw_neighbour(i, j, k, n))
        {
            float f = lateral_rate(sc, hc, soil[n], gw[n].x, gw_inv_d2(k));
            net += f * (f > 0.0f ? a_c : asfloat(gw_aux[n].x));
        }
    }
    gw_aux[c].y = asuint(-step_dt() * net / sc.z);
}

// New depth after removing d >= 0 from h (d <= h) or adding -d; a removal keeps the velocity
float3 change_depth(float3 q, float hn)
{
    float h = q.x;
    float r = hn < h && h > 0.0f ? hn / h : 1.0f;
    return fix_dry(float3(hn, q.y * r, q.z * r));
}

// Vertical terms of a cell, after the last Euler stage, full grid (rain falls and exfiltration comes out on dry
// tiles too, which are then stamped for the next step's active list):
//  1. infiltration / rain chronologies: h += (infiltration rate + rain rate) dt, at most all the water removed;
//  2. groundwater: h_gw += lateral change, then the exchange with the surface water.
// The new state also feeds the s_max and aquifer diffusivity accumulators of the next time step.
[shader("compute")]
GRID_THREADS
void vertical_step(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    uint  j = tid.x, i = tid.y;
    float s = 0.0f, d_gw = 0.0f;
    bool  wet = false;
    if (i < P.nx && j < P.ny)
    {
        uint   c  = idx(i, j);
        float  dt = step_dt();
        float4 q4 = Q[c];
        float3 q  = q4.xyz;
        bool   changed = false;

        if (sources_on())
        {
            float2 rate = source_rates(c);
            float  add  = (rate.x + rate.y) * dt;
            if (add != 0.0f)
            {
                float h  = q.x;
                float hn = add <= -h ? 0.0f : h + add;
                q        = change_depth(q, hn);
                uint4 cum = src_cum[c];
                src_cum[c] = uint4(fixed_add_signed(cum.xy, hn - h), fixed_add_signed(cum.zw, rate.y * dt));
                changed    = true;
            }
        }

        if (gw_on())
        {
            float4 sl  = soil[c];
            float4 g   = gw[c];
            uint4  aux = gw_aux[c];
            uint2  hf  = aux.zw;        // h_gw in fixed point (groundwater.slang fixed_add)
            if (gw_lateral())
                hf = fixed_add(hf, asfloat(aux.y));
            g.x     = fixed_to_float(hf);
            float d = vertical_exchange(sl, g.x, q.x, q4.w, dt);
            if (d != 0.0f)
            {
                // the aquifer receives exactly what the surface lost (h - hn, exact difference)
                float h  = q.x;
                float hn = d >= h ? 0.0f : h - d;        // d <= h: never below 0
                float dv = h - hn;
                q  = change_depth(q, hn);
                hf = fixed_add(hf, dv / sl.z);
                g.x = fixed_to_float(hf);
                if (dv > 0.0f)
                    g.z += dv;
                else
                    g.w -= dv;
                changed = true;
            }
            g.y   = d / dt;
            gw[c] = g;
            gw_aux[c].zw = hf;
            d_gw  = gw_diffusivity(sl, g.x);
        }

        if (changed)
            Q[c] = float4(q, q4.w);
        s   = cell_wave_speed(q);
        wet = is_wet(q);
    }
    accumulate_smax(s);
    if (gw_lateral())
        accumulate_gw_diffusivity(d_gw);
    stamp_wet(wet, gid.y * tiles_y() + gid.x, scalars[S_NSTEPS]);
}

// ---------------------------------------------------------------------------------------------
// Diagnostics and probes (once per frame, in the readback command buffer)
// ---------------------------------------------------------------------------------------------
groupshared float4 g_diag[256];

// x = volume (sum h dx dy), y = wet cells (h > 2e-3), z = max h, w = max speed (h >= h_dry)
float4 diag_combine(float4 a, float4 b)
{
    return float4(a.x + b.x, a.y + b.y, max(a.z, b.z), max(a.w, b.w));
}

void diag_reduce(uint t)
{
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
            g_diag[t] = diag_combine(g_diag[t], g_diag[t + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
}

// Groundwater, sums only: aquifer volume, cumulated infiltration and exfiltration (m3), exchange rate (m3/s)
void diag_gw_reduce(uint t)
{
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
            g_diag[t] += g_diag[t + stride];
        GroupMemoryBarrierWithGroupSync();
    }
}

uint diag_groups_x() { return (P.ny + 15) / 16; }
uint diag_groups()   { return diag_groups_x() * ((P.nx + 15) / 16); }

[shader("compute")]
GRID_THREADS
void diag_partial(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint   j = tid.x, i = tid.y;
    float4 d = float4(0.0f);
    if (i < P.nx && j < P.ny)
    {
        float4 q = Q[idx(i, j)];
        float  h = q.x;
        d.x = max(h, 0.0f) * P.dx * P.dy;
        d.y = h > 2e-3f ? 1.0f : 0.0f;
        d.z = h;
        if (h >= P.h_dry)
        {
            float u = velocity(h, q.y), v = velocity(h, q.z);
            d.w = sqrt(u * u + v * v);
        }
    }
    g_diag[gi] = d;
    diag_reduce(gi);
    if (gi == 0)
        diag[gid.y * diag_groups_x() + gid.x] = g_diag[0];

    if (gw_on())        // uniform: the barriers of the reduction stay in uniform control flow
    {
        GroupMemoryBarrierWithGroupSync();
        float4 e = float4(0.0f);
        if (i < P.nx && j < P.ny)
        {
            uint   c = idx(i, j);
            float4 g = gw[c];
            e = float4(soil[c].z * g.x, g.z, g.w, g.y) * (P.dx * P.dy);
        }
        g_diag[gi] = e;
        diag_gw_reduce(gi);
        if (gi == 0)
            diag_gw[2 * (gid.y * diag_groups_x() + gid.x)] = g_diag[0];
    }
    if (sources_on())
    {
        GroupMemoryBarrierWithGroupSync();
        float4 e = float4(0.0f);
        if (i < P.nx && j < P.ny)
            e = float4(source_cumul(idx(i, j)), 0.0f, 0.0f) * (P.dx * P.dy);
        g_diag[gi] = e;
        diag_gw_reduce(gi);
        if (gi == 0)
            diag_gw[2 * (gid.y * diag_groups_x() + gid.x) + 1] = g_diag[0];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void diag_final(uint gi : SV_GroupIndex)
{
    float4 acc = float4(0.0f);
    uint   n   = diag_groups();
    for (uint k = gi; k < n; k += 256)
        acc = diag_combine(acc, diag[k]);
    g_diag[gi] = acc;
    diag_reduce(gi);
    if (gi == 0)
    {
        scalars[S_VOLUME]    = asuint(g_diag[0].x);
        scalars[S_WET_CELLS] = asuint(g_diag[0].y);
        scalars[S_HMAX]      = asuint(g_diag[0].z);
        scalars[S_UMAX]      = asuint(g_diag[0].w);
    }

    if (gw_on())
    {
        GroupMemoryBarrierWithGroupSync();
        float4 sum = float4(0.0f);
        for (uint k = gi; k < n; k += 256)
            sum += diag_gw[2 * k];
        g_diag[gi] = sum;
        diag_gw_reduce(gi);
        if (gi == 0)
        {
            scalars[S_GW_VOLUME] = asuint(g_diag[0].x);
            scalars[S_GW_INFIL]  = asuint(g_diag[0].y);
            scalars[S_GW_EXFIL]  = asuint(g_diag[0].z);
            scalars[S_GW_RATE]   = asuint(g_diag[0].w);
        }
    }
    if (sources_on())
    {
        GroupMemoryBarrierWithGroupSync();
        float4 sum = float4(0.0f);
        for (uint k = gi; k < n; k += 256)
            sum += diag_gw[2 * k + 1];
        g_diag[gi] = sum;
        diag_gw_reduce(gi);
        if (gi == 0)
        {
            scalars[S_SRC_NET]  = asuint(g_diag[0].x);
            scalars[S_SRC_RAIN] = asuint(g_diag[0].y);
        }
    }
}

[shader("compute")]
[numthreads(8, 1, 1)]
void probes(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= min(probe_cells[0], MAX_PROBES))
        return;
    uint   c = probe_cells[1 + p];
    float4 q = Q[c];
    float  h = q.x;
    uint   s = S_PROBES + 4 * p;
    scalars[s + 0] = asuint(h);
    scalars[s + 1] = asuint(velocity(h, q.y));
    scalars[s + 2] = asuint(velocity(h, q.z));
    scalars[s + 3] = asuint(zb[c] + h);
    if (gw_on())
    {
        float4 g = gw[c];
        scalars[S_GW_PROBES + 2 * p]     = asuint(soil[c].w + g.x);
        scalars[S_GW_PROBES + 2 * p + 1] = asuint(g.y);
    }
}

// Profile points: ground, depth, water table and substratum (ground for both without an aquifer)
[shader("compute")]
[numthreads(64, 1, 1)]
void profile(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= min(profile_cells[0], MAX_PROFILE))
        return;
    uint  c = profile_cells[1 + p];
    float z = zb[c];
    float H = z, zs = z;
    if (gw_on())
    {
        zs = soil[c].w;
        H  = zs + gw[c].x;
    }
    profile_out[p] = float4(z, Q[c].x, H, zs);
}

// ---------------------------------------------------------------------------------------------
// Interactive editing: water block in a cell rectangle (push constants, see WATER_* in params.slang)
// ---------------------------------------------------------------------------------------------
[shader("compute")]
GRID_THREADS
void set_water(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint i0 = uint(stage_coef(0)), j0 = uint(stage_coef(1)), i1 = uint(stage_coef(2)), j1 = uint(stage_coef(3));
    if (i < i0 || i > i1 || j < j0 || j > j1)
        return;
    uint   c     = idx(i, j);
    float  value = stage_coef(4);
    float4 q     = Q[c];
    float  h     = q.x;
    switch (pc.mode)
    {
    case WATER_DEPTH: h = value; break;
    case WATER_LEVEL: h = max(value - q.w, 0.0f); break;
    case WATER_ADD:   h = max(h + value, 0.0f); break;
    default:          h = 0.0f; break;
    }
    // water placed at rest
    Q[c] = float4(h, 0.0f, 0.0f, q.w);
}
