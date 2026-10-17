// Port of the effect kernels of ui/renderer.py: shore mask, relief lighting (cast shadows + ambient
// occlusion), effects carried by the flow (foam, turbidity, wet memory, flood extent) and streamlines.
// Pipeline layout: set 0 = solver set (Q, zb, P, scalars), set 1 = effects set, push constants EffectsPC.
import params;
import reconstruction;
import effects_data;
import noise;

struct EffectsPC
{
    float z_scale;      // lighting
    float frame_dt;     // streamlines: real time of the frame (0 when paused)
    uint  n_sub;        // advection: number of substeps this frame
    uint  sub;          // advection: substep index
    uint  seed;         // random stream
    uint  reset;        // streamlines: respawn everything
    uint  _pad0;
    uint  _pad1;
};
[[vk::push_constant]] ConstantBuffer<EffectsPC> fx;

#define GRID_THREADS [numthreads(16, 16, 1)]
#define GRID_GUARD   uint j = tid.x, i = tid.y; if (i >= P.nx || j >= P.ny) return; uint c = idx(i, j);

// sun at ~40 deg (renderer.py _sun), horizontal direction in physical (x, y) and slope of its rays
static const float3 SUN_RAW = float3(-0.55f, 0.6f, -0.45f);

uint2 clamp_cell(int i, int j)
{
    return uint2(clamp(i, 0, int(P.nx) - 1), clamp(j, 0, int(P.ny) - 1));
}

// bilinear interpolation between cell centres (physical coordinates)
struct Bilinear { uint i, j; float tx, ty; };
Bilinear bilinear(float x, float y)
{
    float fx = clamp(x / P.dx - 0.5f, 0.0f, float(P.nx) - 1.001f);
    float fy = clamp(y / P.dy - 0.5f, 0.0f, float(P.ny) - 1.001f);
    Bilinear b;
    b.i  = uint(fx);
    b.j  = uint(fy);
    b.tx = fx - b.i;
    b.ty = fy - b.j;
    return b;
}

float sample_zb(float x, float y)
{
    Bilinear b = bilinear(x, y);
    return lerp(lerp(zb[idx(b.i, b.j)], zb[idx(b.i + 1, b.j)], b.tx),
                lerp(zb[idx(b.i, b.j + 1)], zb[idx(b.i + 1, b.j + 1)], b.tx), b.ty);
}

float sample_field(RWStructuredBuffer<float> f, float x, float y)
{
    Bilinear b = bilinear(x, y);
    return lerp(lerp(f[idx(b.i, b.j)], f[idx(b.i + 1, b.j)], b.tx),
                lerp(f[idx(b.i, b.j + 1)], f[idx(b.i + 1, b.j + 1)], b.tx), b.ty);
}

// (h, u, v, zb) interpolated between cell centres
float4 sample_state(float x, float y)
{
    Bilinear b = bilinear(x, y);
    float4   r = 0.0f;
    for (uint a = 0; a < 2; ++a)
        for (uint d = 0; d < 2; ++d)
        {
            float  w = (a == 1 ? b.tx : 1.0f - b.tx) * (d == 1 ? b.ty : 1.0f - b.ty);
            uint   k = idx(b.i + a, b.j + d);
            float4 q = Q[k];
            r += w * float4(q.x, velocity(q.x, q.y), velocity(q.x, q.z), zb[k]);
        }
    return r;
}

// PCG hash -> uniform [0, 1)
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
struct Rng
{
    uint s;
    [mutating] float next()
    {
        s = pcg(s);
        return float(s >> 8) * (1.0f / 16777216.0f);
    }
};

// ---------------------------------------------------------------------------------------------
// Shore: wet mask + separable box blur -> sand band around the water, foam on the edge
// ---------------------------------------------------------------------------------------------
[shader("compute")] GRID_THREADS
void compute_wet(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    wet[c] = Q[c].x > 2e-3f ? 1.0f : 0.0f;
}

[shader("compute")] GRID_THREADS
void blur_x(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    int   r = int(E.shore_radius);
    float s = 0.0f;
    for (int k = -r; k <= r; ++k)
    {
        uint2 q = clamp_cell(int(i) + k, int(j));
        s += wet[idx(q.x, q.y)];
    }
    wet_tmp[c] = s / float(2 * r + 1);
}

[shader("compute")] GRID_THREADS
void blur_y(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    int   r = int(E.shore_radius);
    float s = 0.0f;
    for (int k = -r; k <= r; ++k)
    {
        uint2 q = clamp_cell(int(i), int(j) + k);
        s += wet_tmp[idx(q.x, q.y)];
    }
    wet_blur[c] = s / float(2 * r + 1);
}

// ---------------------------------------------------------------------------------------------
// Relief lighting, on the exaggerated relief (as displayed): soft cast shadows + ambient occlusion.
// Recomputed only when the vertical exaggeration changes.
// ---------------------------------------------------------------------------------------------
[shader("compute")] GRID_THREADS
void compute_lighting(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    float3 sun  = normalize(SUN_RAW);
    float  hn   = sqrt(sun.x * sun.x + sun.z * sun.z);
    float  sdx = sun.x / hn, sdy = -sun.z / hn, stan = sun.y / hn;        // display Z = -y
    float  zs   = fx.z_scale;
    float  L    = E.L_ref;

    float x0 = (float(i) + 0.5f) * P.dx, y0 = (float(j) + 0.5f) * P.dy;
    float z0 = zb[c] * zs;

    // soft shadow: march along the ray towards the sun, visibility = smallest angular margin
    float vis = 1.0f, dist = 0.0f, stp = L;
    for (int k = 0; k < 400; ++k)
    {
        dist += stp;
        stp  *= 1.03f;
        float x = x0 + sdx * dist, y = y0 + sdy * dist;
        float ray = z0 + stan * dist;
        if (x < 0.0f || x > E.Lx || y < 0.0f || y > E.Ly || ray > E.z_top * zs)
            break;
        float margin = (ray - sample_zb(x, y) * zs) / dist;
        vis = min(vis, sstep(-0.03f, 0.03f, margin));
    }
    shadow[c] = 0.35f + 0.65f * vis;        // sky light remains in the shade

    // occlusion: elevation of the horizon in 8 directions, at growing distances
    static const float radii[6] = { 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f };
    float occ = 0.0f;
    for (int d = 0; d < 8; ++d)
    {
        float a = float(d) * 0.78539816f;
        float ca = cos(a), sa = sin(a);
        float best = 0.0f;
        for (int r = 0; r < 6; ++r)
        {
            float rr = radii[r] * L;
            best = max(best, (sample_zb(x0 + ca * rr, y0 + sa * rr) * zs - z0) / rr);
        }
        occ += best / sqrt(1.0f + best * best);        // sine of the horizon angle
    }
    ao[c] = 1.0f - 0.9f * occ / 8.0f;
}

// ---------------------------------------------------------------------------------------------
// Effects carried by the flow (semi-Lagrangian advection, in simulated time)
// ---------------------------------------------------------------------------------------------

// Once per frame: simulated time elapsed since the previous frame (0 when paused or after a reset)
[shader("compute")] [numthreads(1, 1, 1)]
void effects_begin()
{
    float t   = asfloat(scalars[S_TIME]);
    float dtf = t - fx_state[0].t_prev;
    fx_state[0].t_frame = dtf > 0.0f ? dtf : 0.0f;
    fx_state[0].t_prev  = t;
}

[shader("compute")] GRID_THREADS
void advect_copy(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    if (fx_state[0].t_frame <= 0.0f)
        return;
    turb_tmp[c] = turb[c];
    foam_tmp[c] = foam[c];
}

[shader("compute")] GRID_THREADS
void advect(uint3 tid : SV_DispatchThreadID)
{
    GRID_GUARD
    float t_frame = fx_state[0].t_frame;
    if (t_frame <= 0.0f)
        return;
    float dt = t_frame / float(fx.n_sub);
    float t  = asfloat(scalars[S_TIME]) - t_frame + float(fx.sub + 1) * dt;
    float L  = E.L_ref;

    float h = Q[c].x;
    if (h > 2e-3f)
    {
        float u = velocity(h, Q[c].y);
        float v = velocity(h, Q[c].z);
        float speed = sqrt(u * u + v * v);
        // semi-Lagrangian advection: fetch the value where the water comes from
        float xc = (float(i) + 0.5f) * P.dx, yc = (float(j) + 0.5f) * P.dy;
        float T = sample_field(turb_tmp, xc - u * dt, yc - v * dt);
        float F = sample_field(foam_tmp, xc - u * dt, yc - v * dt);
        // dissipation: with the distance travelled (+ settling / bursting in calm water)
        T *= exp(-dt * (speed / E.turb_length + 1.0f / E.settle_time));
        F *= exp(-dt * (speed + 0.2f) / E.foam_length);

        // foam sources: supercritical flow, hydraulic jump (supercritical upstream, subcritical here),
        // wave front (relative depth variation from one cell to the next)
        float froude = speed / sqrt(9.81f * max(h, 1e-3f));
        float ex = u / max(speed, 1e-6f), ey = v / max(speed, 1e-6f);
        float4 qu   = sample_state(xc - ex * P.dx, yc - ey * P.dy);
        float fr_up = sqrt(qu.y * qu.y + qu.z * qu.z) / sqrt(9.81f * max(qu.x, 1e-3f));
        float jump  = sstep(1.0f, 1.5f, fr_up) * sstep(1.0f, 0.7f, froude) * sstep(0.05f, 0.3f, speed);
        uint2 ip = clamp_cell(int(i) + 1, int(j)), im = clamp_cell(int(i) - 1, int(j));
        uint2 jp = clamp_cell(int(i), int(j) + 1), jm = clamp_cell(int(i), int(j) - 1);
        float dhx = Q[idx(ip.x, ip.y)].x - Q[idx(im.x, im.y)].x;
        float dhy = Q[idx(jp.x, jp.y)].x - Q[idx(jm.x, jm.y)].x;
        float gh    = sqrt(dhx * dhx + dhy * dhy);
        float front = sstep(0.25f, 0.6f, gh / (2.0f * max(h, 0.05f)));
        float src   = max(max(0.7f * sstep(1.2f, 3.0f, froude), jump), front);
        // intermittent injection (a puff every ~2 cells travelled) -> streaks
        float phase = frac(floor(t * max(speed, 0.05f) / (2.0f * L)) * 0.618034f) * 100.0f;
        float inj   = sstep(0.35f, 0.75f, hash2(float(i) * 0.731f + phase, float(j) * 1.113f - phase));
        foam[c]    = min(max(F, src * inj), 1.0f);
        // turbidity: fast water gets loaded, then the load is carried downstream
        turb[c]    = max(T, sstep(0.3f, 1.5f, speed));
        wet_mem[c] = 1.0f;
        flooded[c] = 1.0f;
    }
    else
    {
        turb[c] = 0.0f;
        foam[c] = 0.0f;
        wet_mem[c] *= exp(-dt / E.dry_time);
    }
}

// ---------------------------------------------------------------------------------------------
// Streamlines: integrated from seeds that drift with the flow
// ---------------------------------------------------------------------------------------------

// seed drawn uniformly in moving water; if no attempt lands in water the seed dies at once and is
// respawned at the next frame
void respawn(uint s, inout Rng rng)
{
    float2 best  = float2(rng.next() * E.Lx, rng.next() * E.Ly);
    bool   found = false;
    for (int k = 0; k < 16 && !found; ++k)
    {
        float2 cand = float2(rng.next() * E.Lx, rng.next() * E.Ly);
        float4 q    = sample_state(cand.x, cand.y);
        if (q.x > 0.02f && sqrt(q.y * q.y + q.z * q.z) > E.u_min_lines)
        {
            best  = cand;
            found = true;
        }
    }
    seeds[s].pos  = best;
    seeds[s].age  = 0.0f;
    seeds[s].life = 1.0f + 1.5f * rng.next();
}

[shader("compute")] [numthreads(64, 1, 1)]
void update_streamlines(uint3 tid : SV_DispatchThreadID)
{
    uint s = tid.x;
    if (s >= N_LINES)
        return;
    Rng rng = { pcg(s * 9781u + fx.seed * 6271u + 1u) };

    if (fx.reset != 0)
    {
        respawn(s, rng);
        seeds[s].age = rng.next() * seeds[s].life;        // staggered ages: no appearance in a block
    }

    // the seed drifts with the flow (simulated time), its life runs in real time
    float  t_frame = fx_state[0].t_frame;
    float4 q       = sample_state(seeds[s].pos.x, seeds[s].pos.y);
    seeds[s].pos  += t_frame * q.yz;
    seeds[s].age  += fx.frame_dt;
    float2 x0 = seeds[s].pos;
    if (seeds[s].age > seeds[s].life || x0.x < 0.0f || x0.x > E.Lx || x0.y < 0.0f || x0.y > E.Ly ||
        q.x < 0.02f || sqrt(q.y * q.y + q.z * q.z) < E.u_min_lines)
        respawn(s, rng);

    // fade in / out
    float a    = seeds[s].age / seeds[s].life;
    float fade = sstep(0.0f, 0.2f, a) * sstep(1.0f, 0.7f, a);

    // instantaneous streamline: follow the velocity direction (RK2, step ds)
    float2 pos   = seeds[s].pos;
    bool   alive = true;
    for (uint m = 0; m < N_PTS; ++m)
    {
        float4 qm = sample_state(pos.x, pos.y);
        float  sp = sqrt(qm.y * qm.y + qm.z * qm.z);
        if (qm.x < 0.01f || sp < E.u_min_lines || pos.x < 0.0f || pos.x > E.Lx || pos.y < 0.0f || pos.y > E.Ly)
            alive = false;
        // white at the head, fading into the water towards the tail (floor: visible in slow areas)
        float tail      = 1.0f - float(m) / float(N_PTS - 1);
        float intensity = fade * tail * (0.35f + 0.65f * sstep(E.u_min_lines, 1.5f, sp));
        LinePoint lp;
        lp.pos       = pos;
        lp.eta       = qm.w + qm.x;
        lp.intensity = alive ? intensity : -1.0f;
        line_pts[s * N_PTS + m] = lp;
        if (alive)
        {
            float2 d1  = qm.yz / max(sp, 1e-6f);
            float2 mid = pos + 0.5f * E.ds * d1;
            float4 qh  = sample_state(mid.x, mid.y);
            float2 d2  = qh.yz;
            pos += E.ds * d2 / max(length(d2), 1e-6f);
        }
    }
}
