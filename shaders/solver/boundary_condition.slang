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

public float inflow_weight(float h)
{
    return pow(max(h, 0.0f), 5.0f / 3.0f);
}

public float inflow_norm(uint side)
{
    return inflow[side];
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
        float q_n = P.bc_value[side] * inflow_weight(q.x) / max(inflow_norm(side), 1e-12f);   // unit discharge (m2/s)
        g = float3(q.x, 0.0f, 0.0f);
        g[AXIS[side]] = INWARD[side] * q_n;
    }
    else    // level
    {
        float hg = max(P.bc_value[side] - z, 0.0f);
        g = float3(hg, hg * velocity(q.x, q.y), hg * velocity(q.x, q.z));
    }
    return g;
}
