// Port of solver/reconstruction.py
// Linear reconstruction (MUSCL, 2nd order) with the Venkatakrishnan limiter.
// h and zb are each limited, then phi_surface = min(phi_h, phi_zb) is applied to both:
// h and zb move together -> the lake at rest is preserved without reconstructing eta.
// Velocities u and v are reconstructed with their own limiter.
module reconstruction;

import params;

// Desingularised velocity (Kurganov & Petrova 2007): u = hm / h when h >> h_dry,
// smoothly tends to 0 in thin films instead of blowing up.
public float velocity(float h, float hm)
{
    float h4 = h * h * h * h;
    return sqrt(2.0f) * h * hm / sqrt(h4 + max(h4, P.h_dry4));
}

// Venkatakrishnan: phi in [0, 1] applied to the half-increment delta = (u_plus - u_minus) / 2
public float flux_limiter(float u_minus, float u_center, float u_plus)
{
    float delta = abs(u_plus - u_minus) * 0.5f;
    float phi   = 1.0f;
    if (delta > 1e-30f)
    {
        float eps2      = P.eps2;
        float d_min     = abs(u_center - min(u_minus, min(u_center, u_plus)));
        float d_max     = abs(max(u_minus, max(u_center, u_plus)) - u_center);
        float ratio_min = (d_min * (d_min + 2.0f * delta) + eps2)
                        / (d_min * (d_min + delta) + 2.0f * delta * delta + eps2);
        float ratio_max = (d_max * (d_max + 2.0f * delta) + eps2)
                        / (d_max * (d_max + delta) + 2.0f * delta * delta + eps2);
        phi = clamp(min(ratio_min, ratio_max), 0.0f, 1.0f);
    }
    return phi;
}

// Face values of one cell along one direction: m = face on the minus side, p = face on the plus side.
// Each is (h, hu, hv, zb).
public struct Recon
{
    public float4 m;
    public float4 p;
};

public Recon reconstruct_linear(float4 q_minus, float4 q_center, float4 q_plus,
                                float zb_minus, float zb_center, float zb_plus)
{
    Recon r;
    r.m = float4(q_center.xyz, zb_center);
    r.p = r.m;
    // constant reconstruction if any of the three cells is dry
    if (q_minus.x >= P.h_eps && q_center.x >= P.h_eps && q_plus.x >= P.h_eps)
    {
        float phi_surface = min(flux_limiter(q_minus.x, q_center.x, q_plus.x),
                                flux_limiter(zb_minus, zb_center, zb_plus));
        // reconstruct velocities (not hu, hv): hu_face = h_face * u_face stays bounded near dry areas
        float u_m = velocity(q_minus.x, q_minus.y), u_c = velocity(q_center.x, q_center.y), u_p = velocity(q_plus.x, q_plus.y);
        float v_m = velocity(q_minus.x, q_minus.z), v_c = velocity(q_center.x, q_center.z), v_p = velocity(q_plus.x, q_plus.z);
        float phi_u = flux_limiter(u_m, u_c, u_p);
        float phi_v = flux_limiter(v_m, v_c, v_p);

        // half step: U_face = U_c +- phi * delta * 0.5, with delta = (U_plus - U_minus) / 2
        float dh  = phi_surface * (q_plus.x - q_minus.x) * 0.25f;
        float dzb = phi_surface * (zb_plus - zb_minus) * 0.25f;
        float du  = phi_u * (u_p - u_m) * 0.25f;
        float dv  = phi_v * (v_p - v_m) * 0.25f;

        // positivity: scale the slope of h AND zb (rather than clipping h on the face),
        // otherwise eta = h + zb is no longer constant on the faces and the lake at rest moves
        if (abs(dh) > q_center.x)
        {
            float scale = q_center.x / abs(dh);
            dh  *= scale;
            dzb *= scale;
        }

        float h_m = max(0.0f, q_center.x - dh);
        float h_p = max(0.0f, q_center.x + dh);
        r.m = float4(h_m, h_m * (u_c - du), h_m * (v_c - dv), zb_center - dzb);
        r.p = float4(h_p, h_p * (u_c + du), h_p * (v_c + dv), zb_center + dzb);
    }
    return r;
}

// m = west face, p = east face of cell (i, j). Boundary cells: constant reconstruction.
public Recon reconstruct_X(uint i, uint j)
{
    uint  c = idx(i, j);
    Recon r;
    if (0 < i && i < P.nx - 1)
        r = reconstruct_linear(Q[idx(i - 1, j)], Q[c], Q[idx(i + 1, j)],
                               zb[idx(i - 1, j)], zb[c], zb[idx(i + 1, j)]);
    else
    {
        r.m = float4(Q[c].xyz, zb[c]);
        r.p = r.m;
    }
    return r;
}

// m = south face, p = north face of cell (i, j). Boundary cells: constant reconstruction.
public Recon reconstruct_Y(uint i, uint j)
{
    uint  c = idx(i, j);
    Recon r;
    if (0 < j && j < P.ny - 1)
        r = reconstruct_linear(Q[idx(i, j - 1)], Q[c], Q[idx(i, j + 1)],
                               zb[idx(i, j - 1)], zb[c], zb[idx(i, j + 1)]);
    else
    {
        r.m = float4(Q[c].xyz, zb[c]);
        r.p = r.m;
    }
    return r;
}
