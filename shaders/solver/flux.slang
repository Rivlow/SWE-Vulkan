// Port of solver/flux.py: HLLC flux + hydrostatic reconstruction (Audusse et al. 2004).
module flux;

import params;
import reconstruction;

public float wave_speed(float h, float u)
{
    return abs(u) + sqrt(P.g * max(h, 0.0f));
}

// Depth "seen" at the interface: h* = max(0, h + z - z_face), velocities kept
public float3 hydrostatic_reconstruct(float3 q, float z, float z_face)
{
    float h  = q.x;
    float u  = velocity(h, q.y);
    float v  = velocity(h, q.z);
    float hs = max(0.0f, h + z - z_face);
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

// HLLC flux in x through a vertical face.
// Mass and normal momentum (hu): HLL flux; tangential momentum (hv): carried by the S* wave.
public float3 Flux_x(float3 q_left, float3 q_right)
{
    float3 f = float3(0.0f);
    float  g = P.g;
    float h_l = q_left.x, hu_l = q_left.y, hv_l = q_left.z;
    float h_r = q_right.x, hu_r = q_right.y, hv_r = q_right.z;
    if (h_l > P.h_eps || h_r > P.h_eps)    // otherwise both sides dry -> zero flux
    {
        float u_l = velocity(h_l, hu_l), v_l = velocity(h_l, hv_l);
        float u_r = velocity(h_r, hu_r), v_r = velocity(h_r, hv_r);
        float3 f_left  = float3(hu_l, hu_l * u_l + 0.5f * g * h_l * h_l, hv_l * u_l);
        float3 f_right = float3(hu_r, hu_r * u_r + 0.5f * g * h_r * h_r, hv_r * u_r);
        float3 s = hllc_wave_speeds(h_l, u_l, h_r, u_r);
        float s_l = s.x, s_r = s.y, s_star = s.z;
        if (s_l >= 0.0f)
            f = f_left;
        else if (s_r <= 0.0f)
            f = f_right;
        else
        {
            f   = (s_r * f_left - s_l * f_right + s_l * s_r * (q_right - q_left)) / (s_r - s_l);
            f.z = f.x * (s_star >= 0.0f ? v_l : v_r);
        }
    }
    return f;
}

// HLLC flux in y through a horizontal face.
// Mass and normal momentum (hv): HLL flux; tangential momentum (hu): carried by the S* wave.
public float3 Flux_y(float3 q_down, float3 q_up)
{
    float3 f = float3(0.0f);
    float  g = P.g;
    float h_d = q_down.x, hu_d = q_down.y, hv_d = q_down.z;
    float h_u = q_up.x, hu_u = q_up.y, hv_u = q_up.z;
    if (h_d > P.h_eps || h_u > P.h_eps)    // otherwise both sides dry -> zero flux
    {
        float u_d = velocity(h_d, hu_d), v_d = velocity(h_d, hv_d);
        float u_u = velocity(h_u, hu_u), v_u = velocity(h_u, hv_u);
        float3 f_down = float3(hv_d, hu_d * v_d, hv_d * v_d + 0.5f * g * h_d * h_d);
        float3 f_up   = float3(hv_u, hu_u * v_u, hv_u * v_u + 0.5f * g * h_u * h_u);
        float3 s = hllc_wave_speeds(h_d, v_d, h_u, v_u);
        float s_d = s.x, s_u = s.y, s_star = s.z;
        if (s_d >= 0.0f)
            f = f_down;
        else if (s_u <= 0.0f)
            f = f_up;
        else
        {
            f   = (s_u * f_down - s_d * f_up + s_d * s_u * (q_up - q_down)) / (s_u - s_d);
            f.y = f.x * (s_star >= 0.0f ? u_d : u_u);
        }
    }
    return f;
}
