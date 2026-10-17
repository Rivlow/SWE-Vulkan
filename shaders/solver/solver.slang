// Port of solver/time_integration.py + compute_flux / apply_BC: one compute entry point per parallel loop.
// The host (src/solver/Solver.cpp) chains them per Runge-Kutta stage with barriers:
//
//   reduce_smax                                   (also copies Q -> Q_old)
//   [finalize_dt] flux_x flux_y source [bc_inflow_norm]
//   bc_fluxes
//   residual -> K[s]
//   combine_fix_dry (stages s > 0) ... final_update
//
// Threads: j = tid.x, i = tid.y so neighbouring threads touch neighbouring addresses (j is fastest).
import params;
import reconstruction;
import flux;
import boundary_condition;

#define GRID_THREADS [numthreads(16, 16, 1)]

// ---------------------------------------------------------------------------------------------
// Time step
// ---------------------------------------------------------------------------------------------

// s_max = max over wet cells (h >= h_dry) of |u|max + sqrt(g h); also stores Q^n into Q_old.
// Positive floats compare like their bit patterns, so a uint atomic max is exact (no float atomics).
[shader("compute")]
GRID_THREADS
void reduce_smax(uint3 tid : SV_DispatchThreadID)
{
    uint  j = tid.x, i = tid.y;
    float s = 0.0f;
    if (i < P.nx && j < P.ny)
    {
        uint   c = idx(i, j);
        float4 q = Q[c];
        Q_old[c] = q;
        float h  = q.x;
        if (h >= P.h_dry)               // thin films do not drive the time step
        {
            float u = velocity(h, q.y);
            float v = velocity(h, q.z);
            s = wave_speed(h, max(abs(u), abs(v)));
        }
    }
    float m = WaveActiveMax(s);
    if (WaveIsFirstLane())
        InterlockedMax(scalars[S_SMAX_BITS], asuint(m));
}

// dt = cfl min(dx, dy) / s_max, t += dt. Resets the accumulator for the next step.
[shader("compute")]
[numthreads(1, 1, 1)]
void finalize_dt()
{
    float s_max = max(asfloat(scalars[S_SMAX_BITS]), 1e-6f);
    float dt    = P.cfl_dx / s_max;
    scalars[S_SMAX_BITS] = 0;
    scalars[S_SMAX]      = asuint(s_max);
    scalars[S_DT]        = asuint(dt);
    scalars[S_TIME]      = asuint(asfloat(scalars[S_TIME]) + dt);
    scalars[S_NSTEPS]    = scalars[S_NSTEPS] + 1;
}

float step_dt() { return asfloat(scalars[S_DT]); }

// ---------------------------------------------------------------------------------------------
// Spatial operator
// ---------------------------------------------------------------------------------------------

// Interior x-faces i in [1, nx-1]: face i lies between cells (i-1, j) and (i, j).
[shader("compute")]
GRID_THREADS
void flux_x(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
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
    F_R[k] = float4(f + float3(0.0f, 0.5f * g * (q_right.x * q_right.x - q_right_star.x * q_right_star.x), 0.0f), 0.0f);
}

// Interior y-faces j in [1, ny-1]: face j lies between cells (i, j-1) and (i, j).
[shader("compute")]
GRID_THREADS
void flux_y(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
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
    G_U[k] = float4(f + float3(0.0f, 0.0f, 0.5f * g * (q_up.x * q_up.x - q_up_star.x * q_up_star.x)), 0.0f);
}

// Centred source term (Audusse 2004, 2nd order): S = g/2 (h_- + h_+)(zb_- - zb_+) from the cell's own
// reconstructed face values; zero when the reconstruction is constant.
[shader("compute")]
GRID_THREADS
void source(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    float g   = P.g;
    Recon r_x = reconstruct_X(i, j);
    Recon r_y = reconstruct_Y(i, j);
    S_b[idx(i, j)] = float2(0.5f * g * (r_x.m.x + r_x.p.x) * (r_x.m.w - r_x.p.w),
                            0.5f * g * (r_y.m.x + r_y.p.x) * (r_y.m.w - r_y.p.w));
}

// Inflow weights sum(h^(5/3) * width) per side. One workgroup per side, fixed-order reduction
// (deterministic, unlike float atomics).
groupshared float g_partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void bc_inflow_norm(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID)
{
    uint side = gid.x;
    uint t    = ltid.x;
    if (P.bc_kind[side] != BC_INFLOW)
        return;                         // uniform per workgroup

    bool  along_y = side < 2;           // west/east sides run along j
    uint  n       = along_y ? P.ny : P.nx;
    float width   = along_y ? P.dy : P.dx;
    float acc     = 0.0f;
    for (uint k = t; k < n; k += 256)
    {
        uint c = side == 0 ? idx(0, k)
               : side == 1 ? idx(P.nx - 1, k)
               : side == 2 ? idx(k, 0)
                           : idx(k, P.ny - 1);
        acc += inflow_weight(Q[c].x) * width;
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
}

// Boundary faces: thread k handles row j = k (west/east) and column i = k (south/north).
[shader("compute")]
[numthreads(256, 1, 1)]
void bc_fluxes(uint3 tid : SV_DispatchThreadID)
{
    uint k = tid.x;
    if (k < P.ny)
    {
        uint j = k;
        // west: cell (0, j) is right of the face
        uint   c = idx(0, j);
        float3 q = Q[c].xyz;
        F_R[fidx(0, j)] = float4(Flux_x(ghost(0, q, zb[c]), q), 0.0f);
        // east: cell (nx-1, j) is left of the face
        c = idx(P.nx - 1, j);
        q = Q[c].xyz;
        F_L[fidx(P.nx, j)] = float4(Flux_x(q, ghost(1, q, zb[c])), 0.0f);
    }
    if (k < P.nx)
    {
        uint i = k;
        // south: cell (i, 0) is above the face
        uint   c = idx(i, 0);
        float3 q = Q[c].xyz;
        G_U[gidx(i, 0)] = float4(Flux_y(ghost(2, q, zb[c]), q), 0.0f);
        // north: cell (i, ny-1) is below the face
        c = idx(i, P.ny - 1);
        q = Q[c].xyz;
        G_D[gidx(i, P.ny)] = float4(Flux_y(q, ghost(3, q, zb[c])), 0.0f);
    }
}

// L(Q) = -(flux balance) / area + source, dQ/dt = L(Q), written to K[pc.stage]
[shader("compute")]
GRID_THREADS
void residual(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c  = idx(i, j);
    float2 sb = S_b[c];
    // the cell is right of its west face (F_R) and left of its east face (F_L)
    float3 L = -(F_L[fidx(i + 1, j)].xyz - F_R[fidx(i, j)].xyz) / P.dx
               - (G_D[gidx(i, j + 1)].xyz - G_U[gidx(i, j)].xyz) / P.dy
               + float3(0.0f, sb.x / P.dx, sb.y / P.dy);
    K[pc.stage * P.n_cells + c] = float4(L, 0.0f);
}

// ---------------------------------------------------------------------------------------------
// Runge-Kutta combinations (pointwise)
// ---------------------------------------------------------------------------------------------

// Q <- Q^n + dt * sum_m coef[m] * K_m   (zero coefficients skipped, like the Python static unroll)
float3 combine(uint c, float dt)
{
    float3 acc = Q_old[c].xyz;
    for (uint m = 0; m < pc.n_coefs; ++m)
    {
        float coef = stage_coef(m);
        if (coef != 0.0f)
            acc += dt * coef * K[m * P.n_cells + c].xyz;
    }
    return acc;
}

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

// Intermediate stage s > 0: Q_s = Q^n + dt sum_j A[s][j] K_j, then fix_dry
[shader("compute")]
GRID_THREADS
void combine_fix_dry(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint c = idx(i, j);
    Q[c] = float4(fix_dry(combine(c, step_dt())), 0.0f);
}

// Q^{n+1} = Q^n + dt sum_s b[s] K_s, fix_dry, friction, outlet cells emptied
[shader("compute")]
GRID_THREADS
void final_update(uint3 tid : SV_DispatchThreadID)
{
    uint j = tid.x, i = tid.y;
    if (i >= P.nx || j >= P.ny)
        return;
    uint   c  = idx(i, j);
    float  dt = step_dt();
    float3 q  = fix_dry(combine(c, dt));
    if ((P.flags & FLAG_FRICTION) != 0)
        q = friction(q, dt);
    if ((P.flags & FLAG_SINK) != 0 && sink[c] != 0)
        q = float3(0.0f);
    Q[c] = float4(q, 0.0f);
}
