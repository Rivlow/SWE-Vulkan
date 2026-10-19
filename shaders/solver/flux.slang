// Port of solver/flux.py: HLLC flux + hydrostatic reconstruction (Audusse et al. 2004).
module flux;

import params;
import reconstruction;

public float wave_speed(float h, float u)
{
    return abs(u) + sqrt(P.g * max(h, 0.0f));
}

// Depth "seen" at the interface: h* = max(0, eta - z_face), velocities kept. eta is the face's free surface as
// reconstructed (h + z mathematically): two faces of a flat surface give the same h* to the bit.
public float3 hydrostatic_reconstruct(float3 q, float eta, float z_face)
{
    float h  = q.x;
    float u  = velocity(h, q.y);
    float v  = velocity(h, q.z);
    float hs = max(0.0f, eta - z_face);
    return float3(hs, hs * u, hs * v);
}

// S_L, S_R (Einfeldt / Toro) and the contact wave S*; u = velocity normal to the face. Dry beds included.
public float3 hllc_wave_speeds(float h_l, float u_l, float h_r, float u_r)
{
    float g   = P.g;
    float c_l = sqrt(g * h_l), c_r = sqrt(g * h_r);
    float s_l = 0.0f, s_r = 0.0f;
    if (h_l <= P.h_eps)                 // dry on the left: dry rarefaction front
    {
        s_l = u_r - 2.0f * c_r;
        s_r = u_r + c_r;
    }
    else if (h_r <= P.h_eps)            // dry on the right
    {
        s_l = u_l - c_l;
        s_r = u_l + 2.0f * c_l;
    }
    else                                // Roe averages
    {
        float sqrt_h_l = sqrt(h_l), sqrt_h_r = sqrt(h_r);
        float u_roe = (sqrt_h_l * u_l + sqrt_h_r * u_r) / (sqrt_h_l + sqrt_h_r);
        float c_roe = sqrt(0.5f * g * (h_l + h_r));
        s_l = min(u_l - c_l, u_roe - c_roe);
        s_r = max(u_r + c_r, u_roe + c_roe);
    }
    float s_star = (s_l * h_r * (u_r - s_r) - s_r * h_l * (u_l - s_l))
                 / min(h_r * (u_r - s_r) - h_l * (u_l - s_l), -1e-12f);   // < 0 as soon as one side is wet
    return float3(s_l, s_r, s_star);
}

// HLLC flux through a face, in deviation form (exact lake at rest, see solver.slang balanced_source).
// The physical flux of a state is its advective part a = (q_n, q_n u_n, q_t u_n) plus the pressure g h^2 / 2 in
// the normal momentum. The face returns the flux minus the pressure of the left state (l) and minus the pressure of
// the right state (r), which is what each cell's balance needs. Pressures never get subtracted from one another:
// their difference is g/2 (h_l + h_r)(h_l - h_r), exactly 0 for equal depths (a subtraction of two g h^2 / 2 can
// be contracted into an fma by the GPU compiler, leaving the rounding error of one of them: 1 ulp of the pressure,
// which made still water move along the domain borders). Same flux as the plain HLLC formula, mathematically.
// States in the face's normal frame: (h, normal discharge, tangential discharge).
// Mass and normal momentum: HLL flux; tangential momentum: carried by the S* wave.
public struct FaceFlux
{
    public float3 l;        // flux - (0, g h_l^2 / 2, 0)
    public float3 r;        // flux - (0, g h_r^2 / 2, 0)
};

public FaceFlux hllc_normal(float3 q_left, float3 q_right)
{
    FaceFlux o;
    o.l = float3(0.0f);
    o.r = float3(0.0f);
    float h_l = q_left.x, h_r = q_right.x;
    if (h_l > P.h_eps || h_r > P.h_eps)    // otherwise both sides dry -> zero flux
    {
        float  u_l = velocity(h_l, q_left.y), v_l = velocity(h_l, q_left.z);
        float  u_r = velocity(h_r, q_right.y), v_r = velocity(h_r, q_right.z);
        float3 a_l = float3(q_left.y, q_left.y * u_l, q_left.z * u_l);
        float3 a_r = float3(q_right.y, q_right.y * u_r, q_right.z * u_r);
        float  dp  = 0.5f * P.g * (h_l + h_r) * (h_l - h_r);        // pressure left - pressure right
        float3 s   = hllc_wave_speeds(h_l, u_l, h_r, u_r);
        float  s_l = s.x, s_r = s.y, s_star = s.z;
        if (s_l >= 0.0f)                    // flux = left state's
        {
            o.l = a_l;
            o.r = a_l + float3(0.0f, dp, 0.0f);
        }
        else if (s_r <= 0.0f)               // flux = right state's
        {
            o.r = a_r;
            o.l = a_r - float3(0.0f, dp, 0.0f);
        }
        else
        {
            float3 fa = (s_r * a_l - s_l * a_r + s_l * s_r * (q_right - q_left)) / (s_r - s_l);
            fa.z      = fa.x * (s_star >= 0.0f ? v_l : v_r);
            o.l = fa + float3(0.0f, s_l * dp / (s_r - s_l), 0.0f);
            o.r = fa + float3(0.0f, s_r * dp / (s_r - s_l), 0.0f);
        }
    }
    return o;
}

// x-faces: the normal frame is (h, hu, hv) itself
public FaceFlux Flux_x_dev(float3 q_left, float3 q_right)
{
    return hllc_normal(q_left, q_right);
}

// y-faces: (h, hv, hu) in the normal frame, mapped back to (h, hu, hv)
public FaceFlux Flux_y_dev(float3 q_down, float3 q_up)
{
    FaceFlux n = hllc_normal(q_down.xzy, q_up.xzy);
    FaceFlux o;
    o.l = n.l.xzy;
    o.r = n.r.xzy;
    return o;
}

// Plain fluxes (the deviation of the left / lower state plus its pressure)
public float3 Flux_x(float3 q_left, float3 q_right)
{
    return Flux_x_dev(q_left, q_right).l + float3(0.0f, pressure(q_left.x), 0.0f);
}
public float3 Flux_y(float3 q_down, float3 q_up)
{
    return Flux_y_dev(q_down, q_up).l + float3(0.0f, 0.0f, pressure(q_down.x));
}
