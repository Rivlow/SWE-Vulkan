// Port of solver/boundary_condition.py
// Boundary conditions imposed as fluxes: HLLC flux between the boundary cell and a ghost state.
// Each side (west, east, south, north) is:
//   BC_WALL     reflective wall: mirror state (normal velocity reversed) -> zero mass flux
//   BC_INFLOW   inflow discharge Q (m3/s) spread along the side in proportion to h^(5/3) (Manning)
//   BC_LEVEL    imposed free-surface level (subcritical outflow), cell velocity kept
// Boundary fluxes are first order (raw cell values), like the Python version.
module boundary_condition;

import params;
import reconstruction;

public static const uint  AXIS[4]   = { 1, 1, 2, 2 };              // momentum component normal to the side
public static const float INWARD[4] = { 1.0f, -1.0f, 1.0f, -1.0f }; // inward direction along that axis

// Effective depth driving the inflow distribution along a side: the actual depth, or, on a dry side, the
// depth the cell would have if the side were filled up to bc_fill above its lowest point (the river channel).
public float inflow_depth(uint side, float h, float z)
{
    float fill = P.bc_zmin[side] + P.bc_fill - z;
    return max(h, max(fill, 0.0f));
}

public float inflow_weight(float h_eff)
{
    return pow(max(h_eff, 0.0f), 5.0f / 3.0f);
}

public float inflow_norm(uint side)
{
    return inflow[side];
}

// Unit discharge (m2/s) entering through the boundary cell of an inflow side
public float inflow_unit_discharge(uint side, float3 q, float z)
{
    float h_eff = inflow_depth(side, q.x, z);
    return P.bc_value[side] * inflow_weight(h_eff) / max(inflow_norm(side), 1e-12f);
}

public float3 ghost(uint side, float3 q, float z)
{
    float3 g = q;
    uint kind = P.bc_kind[side];
    if (kind == BC_WALL)
    {
        g[AXIS[side]] = -q[AXIS[side]];
    }
    else if (kind == BC_INFLOW)
    {
        float h_eff = inflow_depth(side, q.x, z);
        float q_n   = inflow_unit_discharge(side, q, z);
        // the ghost must be wet for the HLLC flux to pass water into a dry cell: use the effective depth
        g = float3(max(q.x, h_eff), 0.0f, 0.0f);
        g[AXIS[side]] = INWARD[side] * q_n;
    }
    else    // level
    {
        float hg = max(P.bc_value[side] - z, 0.0f);
        g = float3(hg, hg * velocity(q.x, q.y), hg * velocity(q.x, q.z));
    }
    return g;
}
