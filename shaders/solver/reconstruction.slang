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

// Venkatakrishnan (1993): phi = (d^2 + 2 d D2 + eps^2) / (d^2 + d D2 + 2 D2^2 + eps^2), d = distance from the
// centre to the max (min) of the stencil, D2 = the increment the reconstruction adds on the face:
// U_face = U_c +- phi (u_plus - u_minus) / 4, so D2 = |u_plus - u_minus| / 4. On linear data d = 2 D2 and phi = 1:
// smooth profiles keep their full slope. (The Python reference passes D2 = |u_plus - u_minus| / 2, the increment
// to the neighbour: phi = 0.75 on linear data, every slope cut by a quarter, which slows a dry front by ~1 m/s on
// the Ritter dam break whatever dx. FLAG_LEGACY_LIMITER keeps it, to compare with the reference.)
public float flux_limiter(float u_minus, float u_center, float u_plus)
{
    float delta = abs(u_plus - u_minus) * ((P.flags & FLAG_LEGACY_LIMITER) != 0 ? 0.5f : 0.25f);
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
// Each is (h, hu, hv, zb), plus the free surface eta of each face.
//
// Exact lake at rest: the free surface is reconstructed itself (eta = h + zb of each cell, limited with the same
// phi as h and zb), and h_face = eta_face - zb_face. Mathematically the same faces as reconstructing h and zb
// (eta's slope is the sum of theirs); in floating point, cells whose eta are equal to the bit give faces with
// exactly that eta, which the flux and source terms (solver.slang) turn into exactly zero.
public struct Recon
{
    public float4 m;
    public float4 p;
    public float  eta_m;
    public float  eta_p;
};

// Hydrostatic pressure term g h^2 / 2. One function for the flux and the balanced source terms: the same
// expression gives the same bits, so a still face cancels exactly.
public float pressure(float h)
{
    return 0.5f * P.g * h * h;
}

// Depth that puts the free surface at `level` exactly in floating point: h + z == level to the bit (h >= 0).
// fl(level - z) is within half an ulp of the exact depth; a step of one ulp of h fixes the rare ties.
public float depth_for_level(float level, float z)
{
    float h = level - z;
    if (h <= 0.0f)
        return 0.0f;
    for (int k = 0; k < 4 && h + z != level; ++k)
        h = asfloat(asuint(h) + (h + z < level ? 1u : 0xffffffffu));
    return max(h, 0.0f);
}

public Recon reconstruct_linear(float4 q_minus, float4 q_center, float4 q_plus,
                                float zb_minus, float zb_center, float zb_plus)
{
    Recon r;
    r.m = float4(q_center.xyz, zb_center);
    r.p = r.m;
    r.eta_m = q_center.x + zb_center;
    r.eta_p = r.eta_m;
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

        // eta's slope: dh + dzb mathematically, computed from the cells' eta so that a flat surface gives 0 exactly
        float eta_c  = q_center.x + zb_center;
        float deta   = phi_surface * ((q_plus.x + zb_plus) - (q_minus.x + zb_minus)) * 0.25f;

        // positivity: scale the slope of h AND zb (rather than clipping h on the face),
        // otherwise eta = h + zb is no longer constant on the faces and the lake at rest moves
        if (abs(dh) > q_center.x)
        {
            float scale = q_center.x / abs(dh);
            dh   *= scale;
            dzb  *= scale;
            deta *= scale;
        }

        // the slope scaling keeps eta >= z on the faces mathematically; max() only catches round-off, and keeps
        // eta_face = h_face + z_face for the balanced source term
        float z_m   = zb_center - dzb, z_p = zb_center + dzb;
        float eta_m = max(eta_c - deta, z_m), eta_p = max(eta_c + deta, z_p);
        float h_m   = eta_m - z_m;
        float h_p   = eta_p - z_p;
        r.m = float4(h_m, h_m * (u_c - du), h_m * (v_c - dv), z_m);
        r.p = float4(h_p, h_p * (u_c + du), h_p * (v_c + dv), z_p);
        r.eta_m = eta_m;
        r.eta_p = eta_p;
    }
    return r;
}

// Solver kernels read the bed from Q.w (one 16-byte load per cell instead of two streams).
Recon reconstruct_cells(float4 q_minus, float4 q_center, float4 q_plus)
{
    return reconstruct_linear(q_minus, q_center, q_plus, q_minus.w, q_center.w, q_plus.w);
}

// m = west face, p = east face of cell (i, j). Boundary cells: constant reconstruction.
public Recon reconstruct_X(uint i, uint j)
{
    float4 q = Q[idx(i, j)];
    Recon  r;
    // bridge cells and their neighbours: constant (the lid changes the pressure law across the stencil)
    if (0 < i && i < P.nx - 1 && !(bridges_on() && (bridge_at(idx(i - 1, j)) || bridge_at(idx(i, j)) || bridge_at(idx(i + 1, j)))))
        r = reconstruct_cells(Q[idx(i - 1, j)], q, Q[idx(i + 1, j)]);
    else
    {
        r.m = q;
        r.p = q;
        r.eta_m = q.x + q.w;
        r.eta_p = r.eta_m;
    }
    return r;
}

// m = south face, p = north face of cell (i, j). Boundary cells: constant reconstruction.
public Recon reconstruct_Y(uint i, uint j)
{
    float4 q = Q[idx(i, j)];
    Recon  r;
    if (0 < j && j < P.ny - 1 && !(bridges_on() && (bridge_at(idx(i, j - 1)) || bridge_at(idx(i, j)) || bridge_at(idx(i, j + 1)))))
        r = reconstruct_cells(Q[idx(i, j - 1)], q, Q[idx(i, j + 1)]);
    else
    {
        r.m = q;
        r.p = q;
        r.eta_m = q.x + q.w;
        r.eta_p = r.eta_m;
    }
    return r;
}
