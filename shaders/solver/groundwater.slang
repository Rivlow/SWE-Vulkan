// Groundwater layer: unconfined aquifer under the surface water, port of wolfgpu's groundwater model.
// Per cell: soil = (kv, kh, Sy, z_sub), gw = (h_gw, exchange rate, cumulated infiltration, cumulated exfiltration).
// Kernels (solver.slang, full grid after the last Euler stage: the aquifer lives under dry tiles too):
//
//   gw_alpha     outflow factor of each cell (lateral flow only)
//   gw_lateral   change of h_gw by the lateral fluxes over the step, into gw_aux.y
//   vertical_step  chronologies, h_gw += lateral change, then the vertical exchange with the surface water
//
// Water table H = z_sub + h_gw, stored water Sy h_gw per unit area.
//
// Vertical exchange: one depth d per step, taken from one layer and given to the other (exact conservation,
// no limiter involved), capped by what each side can give or take:
//   infiltration  zb + h > H:  d =  min(kv h / (h_gw + 1 m) dt, h, Sy (zb - H))    soil room above the table
//   exfiltration  H > zb + h:  d = -min(kv dt, Sy (H - zb))                         table above the ground
// The infiltration rate grows with the ponding depth h and drops as the aquifer fills (h_gw); no Green-Ampt yet.
// wolfgpu convention (FLAG_GW_WOLFGPU, Sy = 1, z_sub = zb, compute_infiltration_flux.comp): the vertical test
// compares zb + h with h_gw itself, no storage limit: d = min(kv dt, h) or d = -min(kv dt, h_gw).
//
// Lateral flow (Dupuit-Boussinesq, wolfgpu compute_gw_flux_at_borders.comp): across a face of length L between
// cells a and b, the discharge is T (H_a - H_b) L / d with the transmissivity T = kh h_gw of the upstream cell
// (higher head). Explicit: stable for dt <= Sy / (4 T (1/dx^2 + 1/dy^2)), folded into the time step.
// Positivity: each face is scaled by the outflow factor of its donor, alpha = min(1, Sy h_gw / (dt O)), O the
// raw outflow (the first Jacobi pass of the surface drain limiter: inflow is not counted, never needed).
module groundwater;

import params;

// infiltration rate kv h / (h_gw + INFILTRATION_LENGTH) (m)
static const float INFILTRATION_LENGTH = 1.0f;

public bool gw_on()      { return (P.flags & FLAG_GROUNDWATER) != 0; }
public bool gw_lateral() { return (P.flags & FLAG_GW_LATERAL) != 0; }

// Depth moved from the surface water to the aquifer over dt (< 0: exfiltration). s = soil of the cell.
public float vertical_exchange(float4 s, float h_gw, float h, float zb, float dt)
{
    float kv = s.x;
    if (kv <= 0.0f)
        return 0.0f;
    float rate = kv * dt;
    if ((P.flags & FLAG_GW_WOLFGPU) != 0)
        return zb + h > h_gw ? min(rate, h) : -min(rate, h_gw);
    float H = s.w + h_gw;
    if (zb + h > H)
        return min(min(rate * h / (h_gw + INFILTRATION_LENGTH), h), max(s.z * (zb - H), 0.0f));
    return -min(rate, max(s.z * (H - zb), 0.0f));
}

// Water-depth rate (m/s over the area of a) leaving a towards b across their face, per unit (H_a - H_b) / d^2:
// T (H_a - H_b) / d^2 with the upstream transmissivity. inv_d2 = 1 / dx^2 or 1 / dy^2.
public float lateral_rate(float4 sa, float ha, float4 sb, float hb, float inv_d2)
{
    float Ha = sa.w + ha, Hb = sb.w + hb;
    float T  = Ha >= Hb ? sa.y * ha : sb.y * hb;
    return T * (Ha - Hb) * inv_d2;
}

// h_gw is accumulated in 64-bit fixed point like the clock (gw_aux.z: units of 2^-8 m, gw_aux.w: units of 2^-40 m);
// gw.x is its float copy for the readers. A slow exchange (1e-8 m per step) is far below the ulp of a float h_gw
// (3e-8 m at 0.3 m) and was lost: water left the surface without reaching the aquifer. Integer additions cannot
// be reassociated by the compiler, unlike a compensated float sum.
public float fixed_to_float(uint2 f)
{
    return float(f.x) * (1.0f / 256.0f) + float(f.y) * 9.094947e-13f;        // 2^-40
}

// f + x, clamped at 0 (an emptied cell)
public uint2 fixed_add(uint2 f, float x)
{
    float ax    = abs(x) * 256.0f;                          // exact (power of two)
    float whole = floor(ax);
    uint  hi    = uint(whole);
    uint  lo    = uint((ax - whole) * 4294967296.0f);       // exact difference, scaled by 2^32
    if (x >= 0.0f)
    {
        uint s = f.y + lo;
        return uint2(f.x + hi + (s < lo ? 1u : 0u), s);
    }
    if (hi > f.x || (hi == f.x && lo > f.y))
        return uint2(0u, 0u);
    return uint2(f.x - hi - (f.y < lo ? 1u : 0u), f.y - lo);
}

// Signed variant (two's complement over the 64 bits, no clamp): cumulated volumes of the chronologies
public uint2 fixed_add_signed(uint2 f, float x)
{
    float ax    = abs(x) * 256.0f;
    float whole = floor(ax);
    uint  hi    = uint(whole);
    uint  lo    = uint((ax - whole) * 4294967296.0f);
    if (x >= 0.0f)
    {
        uint s = f.y + lo;
        return uint2(f.x + hi + (s < lo ? 1u : 0u), s);
    }
    return uint2(f.x - hi - (f.y < lo ? 1u : 0u), f.y - lo);
}

public float fixed_to_float_signed(uint2 f)
{
    if ((f.x & 0x80000000u) == 0)
        return fixed_to_float(f);
    // negative: magnitude = two's complement
    uint lo = ~f.y + 1u;
    uint hi = ~f.x + (lo == 0u ? 1u : 0u);
    return -fixed_to_float(uint2(hi, lo));
}

// Cumulated depths of the chronologies of cell c: net depth added to the surface, rain depth (m)
public float2 source_cumul(uint c)
{
    uint4 s = src_cum[c];
    return float2(fixed_to_float_signed(s.xy), fixed_to_float_signed(s.zw));
}

// Diffusivity of the aquifer at a cell (m2/s): drives the explicit stability limit
public float gw_diffusivity(float4 s, float h_gw)
{
    return s.y * max(h_gw, 0.0f) / s.z;
}

// The four neighbours of (i, j): west, east, south, north; false across the domain boundary (no flow)
public bool gw_neighbour(uint i, uint j, uint k, out uint n)
{
    n = 0;
    if (k == 0) { if (i == 0)        return false; n = idx(i - 1, j); }
    if (k == 1) { if (i + 1 >= P.nx) return false; n = idx(i + 1, j); }
    if (k == 2) { if (j == 0)        return false; n = idx(i, j - 1); }
    if (k == 3) { if (j + 1 >= P.ny) return false; n = idx(i, j + 1); }
    return true;
}

public float gw_inv_d2(uint k)
{
    return k < 2 ? 1.0f / (P.dx * P.dx) : 1.0f / (P.dy * P.dy);
}

// ---------------------------------------------------------------------------------------------
// Infiltration / rain chronologies (wolfgpu update_globals.comp + compute_sources.comp)
// ---------------------------------------------------------------------------------------------
// Zone maps and tables of rates over time. The rates of the step are looked up once per step (finalize_dt, one
// thread, at the time of the start of the step), then every cell adds (infiltration rate + rain rate) dt to its
// depth in the vertical kernel; a negative total removes at most the water present.

public bool sources_on() { return (P.flags & FLAG_SOURCES) != 0; }

// Table 0 (infiltration) or 1 (rain): rates of its zones at time t into src_rates. No line started yet: 0; past
// the last line: the last line; between two lines: the first one, or linear interpolation (lerp).
public void chrono_rates(uint table, float t)
{
    uint h     = 4 * table;
    uint lines = uint(chrono[h]), nz = uint(chrono[h + 1]), off = uint(chrono[h + 3]);
    bool lerp_ = chrono[h + 2] != 0.0f;
    uint base  = table == 0 ? 0 : uint(chrono[1]);        // rain rates follow the infiltration ones
    uint row   = 1 + nz;
    int  line  = -1;
    for (uint l = 0; l < lines; ++l)
    {
        if (t < chrono[off + l * row])
            break;
        line = int(l);
    }
    for (uint z = 0; z < nz; ++z)
    {
        float q = 0.0f;
        if (line >= 0)
        {
            uint r = off + uint(line) * row;
            q = chrono[r + 1 + z];
            if (lerp_ && uint(line) + 1 < lines)
            {
                float t0 = chrono[r], t1 = chrono[r + row];
                float a  = (t - t0) / (t1 - t0);
                q = q * (1.0f - a) + chrono[r + row + 1 + z] * a;
            }
        }
        src_rates[base + z] = q;
    }
}

// (infiltration zone rate, rain rate) of cell c this step (m/s, > 0 adds water to the surface)
public float2 source_rates(uint c)
{
    uint  zz = src_zones[c];
    uint  zi = zz & 0xffffu, zr = zz >> 16;
    float ri = zi != 0 ? src_rates[zi - 1] : 0.0f;
    float rr = zr != 0 ? src_rates[uint(chrono[1]) + zr - 1] : 0.0f;
    return float2(ri, rr);
}
