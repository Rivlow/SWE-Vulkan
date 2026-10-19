// Bridges: flow under a deck, pressurised flow when the water reaches the soffit, overtopping above the deck.
//
// A bridge cell has a soffit (underside of the deck) and a deck top: the clearance above the bed c = z_soffit - zb
// and the deck thickness d = z_top - z_soffit. The state keeps h as the piezometric depth (eta = zb + h is the free
// surface, or the pressure head under the deck), so that the reconstruction, the boundary conditions, the zones and
// the rendering see the usual variables. Three regimes:
//   h <= c           free surface under the deck
//   c < h <= c + d   pressurised: the opening is full, h - c is the pressure head above the soffit
//   h > c + d        overtopping: the opening plus a layer of h - c - d over the deck
// Model (one momentum per cell, rigid lid):
//   flowing depth   h_eff(h) = min(h, c) + max(h - c - d, 0)       velocity u = q / h_eff
//   stored depth    s(h)     = h | c + e (h - c) | c + e d + (h - c - d)   (Preissmann slot of relative width e)
//   pressure        dP/dh    = g h_eff  -> the force -g h_eff grad(eta) on the flowing section
//   celerity        sqrt(g h_eff / s'(h)): sqrt(g c / e) under pressure (the slot keeps it finite)
// The mass balance is written on s (the deck holds no water, the slot holds at most e d per unit area), so the
// physical volume is sum(s) dx dy; the depth follows as h = s^-1. e = 1 gives no slot (wolfgpu's model: the
// pressurised cell stores the whole piezometric column, its waves travel at sqrt(g h)).
// A face between two cells takes the lowest soffit and the lowest deck top of its two cells (the opening of the
// bridge); a soffit at the bed makes the deck a weir (only the overtopping flows).
module bridge;

import params;
import reconstruction;

public struct Lid
{
    public float c;        // clearance above the bed (m), >= 0; huge where there is no bridge
    public float d;        // deck thickness (m), >= 0
};

// Lid of a cell or a face at bed z from (soffit, top) elevations
public Lid lid_at(float2 b, float z)
{
    Lid l;
    l.c = max(b.x - z, 0.0f);
    l.d = b.x >= NO_BRIDGE ? 0.0f : max(b.y - b.x, 0.0f);
    return l;
}

public Lid cell_lid(uint c, float z)
{
    return lid_at(bridge_at(c) ? bridge_lid[c] : float2(NO_BRIDGE, NO_BRIDGE), z);
}

// Opening of a face between cells a and b, at the face bed z_face
public Lid face_lid(uint a, uint b, float z_face)
{
    float2 ba = bridge_at(a) ? bridge_lid[a] : float2(NO_BRIDGE, NO_BRIDGE);
    float2 bb = bridge_at(b) ? bridge_lid[b] : float2(NO_BRIDGE, NO_BRIDGE);
    return lid_at(float2(min(ba.x, bb.x), min(ba.y, bb.y)), z_face);
}

public float h_eff(float h, Lid l)
{
    return min(h, l.c) + max(h - l.c - l.d, 0.0f);
}

public float stored(float h, Lid l)
{
    if (h <= l.c)
        return h;
    float e = P.bridge_slot;
    if (h <= l.c + l.d)
        return l.c + e * (h - l.c);
    return l.c + e * l.d + (h - l.c - l.d);
}

public float depth_of_stored(float s, Lid l)
{
    if (s <= l.c)
        return s;
    float e  = P.bridge_slot;
    float sp = l.c + e * l.d;        // stored when the opening is full up to the deck top
    if (s <= sp)
        return l.c + (s - l.c) / e;
    return l.c + l.d + (s - sp);
}

// Depth for a stored depth s, given a reference state (h_ref, its stored depth m_ref): h_ref itself when s did not
// change (the round trip h -> s -> h is not exact in float: under pressure (s - c) / e magnifies the rounding 1/e
// times, still water would move by ulps)
public float depth_from(float s, float h_ref, float m_ref, Lid l)
{
    return s == m_ref ? h_ref : depth_of_stored(s, l);
}

public float storage_slope(float h, Lid l)
{
    return (h > l.c && h <= l.c + l.d) ? P.bridge_slot : 1.0f;
}

// Celerity of small waves
public float bridge_celerity(float h, Lid l)
{
    return sqrt(P.g * max(h_eff(h, l), 0.0f) / storage_slope(h, l));
}

// Stored depth of a cell (its own lid), h itself without a bridge
public float stored_at(uint c, float h, float z)
{
    return bridge_at(c) ? stored(h, cell_lid(c, z)) : h;
}

// HLLC flux through a face with an opening `l`, in the face's normal frame (h piezometric at the face, normal
// discharge, tangential discharge), in deviation form like hllc_normal (flux.slang): l / r = flux minus the
// pressure of the left / right state, pressures differing by g/2 (h_eff_l + h_eff_r)(h_l - h_r) (dP/dh = g h_eff).
// Mass is conserved in the stored depth s: the HLL dissipation acts on (s, q_n, q_t). Without a lid (c huge) every
// expression is the plain one, but bridge-free faces keep using hllc_normal (same bits as before).
public struct BridgeFlux
{
    public float3 l;
    public float3 r;
};

public BridgeFlux hllc_bridge(float3 q_left, float3 q_right, Lid lid)
{
    BridgeFlux o;
    o.l = float3(0.0f);
    o.r = float3(0.0f);
    float h_l = q_left.x, h_r = q_right.x;
    float e_l = h_eff(h_l, lid), e_r = h_eff(h_r, lid);
    if (e_l > P.h_eps || e_r > P.h_eps)        // otherwise nothing can flow through the opening
    {
        float  u_l = velocity(e_l, q_left.y), v_l = velocity(e_l, q_left.z);
        float  u_r = velocity(e_r, q_right.y), v_r = velocity(e_r, q_right.z);
        float3 a_l = float3(q_left.y, q_left.y * u_l, q_left.z * u_l);
        float3 a_r = float3(q_right.y, q_right.y * u_r, q_right.z * u_r);
        float  dp  = 0.5f * P.g * (e_l + e_r) * (h_l - h_r);
        float  c_l = bridge_celerity(h_l, lid), c_r = bridge_celerity(h_r, lid);
        float  s_l, s_r;
        if (e_l <= P.h_eps)
        {
            s_l = u_r - 2.0f * c_r;
            s_r = u_r + c_r;
        }
        else if (e_r <= P.h_eps)
        {
            s_l = u_l - c_l;
            s_r = u_l + 2.0f * c_l;
        }
        else
        {
            float sq_l = sqrt(e_l), sq_r = sqrt(e_r);
            float u_roe = (sq_l * u_l + sq_r * u_r) / (sq_l + sq_r);
            float c_roe = sqrt(0.5f * (c_l * c_l + c_r * c_r));
            s_l = min(u_l - c_l, u_roe - c_roe);
            s_r = max(u_r + c_r, u_roe + c_roe);
        }
        float m_l = stored(h_l, lid), m_r = stored(h_r, lid);
        float s_star = (s_l * m_r * (u_r - s_r) - s_r * m_l * (u_l - s_l))
                     / min(m_r * (u_r - s_r) - m_l * (u_l - s_l), -1e-12f);
        if (s_l >= 0.0f)
        {
            o.l = a_l;
            o.r = a_l + float3(0.0f, dp, 0.0f);
        }
        else if (s_r <= 0.0f)
        {
            o.r = a_r;
            o.l = a_r - float3(0.0f, dp, 0.0f);
        }
        else
        {
            float3 U_l = float3(m_l, q_left.y, q_left.z), U_r = float3(m_r, q_right.y, q_right.z);
            float3 fa  = (s_r * a_l - s_l * a_r + s_l * s_r * (U_r - U_l)) / (s_r - s_l);
            fa.z       = fa.x * (s_star >= 0.0f ? v_l : v_r);
            o.l = fa + float3(0.0f, s_l * dp / (s_r - s_l), 0.0f);
            o.r = fa + float3(0.0f, s_r * dp / (s_r - s_l), 0.0f);
        }
    }
    return o;
}

// Face state of a cell for a bridge face: piezometric depth h* = max(0, eta - z_face) and the discharge entering the
// opening. A bed step reduces it like the hydrostatic reconstruction (q h* / h); the lid does not: the discharge goes
// through the smaller opening faster (keeping the velocity instead cut the discharge by c / h at the deck entrance,
// 5 -> 4.1 m2/s for c = 5 m under 6 m, and the Riemann solver answered with grid-scale oscillations). A closed
// opening (h_eff = 0 at the face) passes nothing.
public float3 bridge_face_state(float3 q, float eta, uint c, float z_cell, float z_face, Lid face)
{
    float hs = max(0.0f, eta - z_face);
    float es = h_eff(hs, face);
    if (es <= P.h_eps || q.x <= 0.0f)
        return float3(hs, 0.0f, 0.0f);
    float step = min(hs / q.x, 1.0f);        // bed step (Audusse)
    return float3(hs, q.y * step, q.z * step);
}

// Manning friction of a bridge cell: hydraulic radius h_eff / n_walls, n_walls = 1 (free surface), 2 (bed and soffit
// wetted, pressurised), 3 (overtopping: + the deck top). Semi-implicit like the open-channel friction.
public float3 bridge_friction(float3 q, float dt, Lid l)
{
    float e = h_eff(q.x, l);
    if (e > P.h_eps)
    {
        float walls = q.x <= l.c ? 1.0f : (q.x <= l.c + l.d ? 2.0f : 3.0f);
        float u  = velocity(e, q.y), v = velocity(e, q.z);
        float R  = e / walls;        // dq/dt = -g h_eff S_f = -g n^2 |u| q / R^(4/3)
        float cf = P.g_n2 * sqrt(u * u + v * v) / pow(R, 4.0f / 3.0f);
        q.y /= 1.0f + dt * cf;
        q.z /= 1.0f + dt * cf;
    }
    return q;
}
