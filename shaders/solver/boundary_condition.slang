// Port of solver/boundary_condition.py
// Boundary conditions imposed as fluxes: HLLC flux between the boundary cell and a ghost state.
// Each side (west, east, south, north) is:
//   BC_WALL     reflective wall: mirror state (normal velocity reversed) -> zero mass flux
//   BC_INFLOW   inflow discharge Q (m3/s) spread along the side in proportion to h^(5/3) (Manning)
//   BC_LEVEL    imposed free-surface level (subcritical outflow), cell velocity kept
//   BC_FREE     free outflow: the ghost repeats the cell (zero gradient), water leaves without reflection
//   BC_DEPTH    imposed water depth, cell velocity kept
// The same ghost states serve the weak conditions on cell faces (solver.slang face_bc_apply).
// Boundary fluxes are first order (raw cell values), like the Python version.
module boundary_condition;

import params;
import reconstruction;

public static const uint  AXIS[4]   = { 1, 1, 2, 2 };              // momentum component normal to the side
public static const float INWARD[4] = { 1.0f, -1.0f, 1.0f, -1.0f }; // inward direction along that axis

// Effective depth driving the inflow distribution: the actual depth, or, on a dry border, the depth the cell
// would have if the border were filled up to bc_fill above its lowest point zmin (the river channel).
public float inflow_depth_z(float zmin, float h, float z)
{
    float fill = zmin + P.bc_fill - z;
    return max(h, max(fill, 0.0f));
}

public float inflow_depth(uint side, float h, float z)
{
    return inflow_depth_z(P.bc_zmin[side], h, z);
}

public float inflow_weight(float h_eff)
{
    return pow(max(h_eff, 0.0f), 5.0f / 3.0f);
}

public float inflow_norm(uint side)
{
    return inflow[side];
}

// Unit discharge (m2/s) entering through a face of an inflow border: total discharge `value` shared in
// proportion to h_eff^(5/3), `norm` = sum of the weights times the face widths over the border
public float unit_discharge(float value, float zmin, float norm, float h, float z)
{
    return value * inflow_weight(inflow_depth_z(zmin, h, z)) / max(norm, 1e-12f);
}

public float inflow_unit_discharge(uint side, float3 q, float z)
{
    return unit_discharge(P.bc_value[side], P.bc_zmin[side], inflow_norm(side), q.x, z);
}

// Ghost state of a border of kind `kind` next to the cell state q (bed z): axis = normal momentum component,
// inward = sign of the inward direction along that axis
public float3 ghost_of(uint kind, float value, float zmin, float norm, uint axis, float inward, float3 q, float z)
{
    float3 g = q;
    if (kind == BC_WALL)
    {
        g[axis] = -q[axis];
    }
    else if (kind == BC_INFLOW)
    {
        float h_eff = inflow_depth_z(zmin, q.x, z);
        float q_n   = unit_discharge(value, zmin, norm, q.x, z);
        // the ghost must be wet for the HLLC flux to pass water into a dry cell: use the effective depth
        g = float3(max(q.x, h_eff), 0.0f, 0.0f);
        g[axis] = inward * q_n;
    }
    else if (kind == BC_FREE)
    {
        g = q;        // zero gradient: the HLLC flux of two equal states is the cell's own flux
    }
    else    // level or depth: the ghost carries the imposed depth with the cell's velocity
    {
        // a level equal to the still water of the cell gives the cell's own depth to the bit (no spurious flux)
        float hg = kind == BC_DEPTH ? max(value, 0.0f) : depth_for_level(value, z);
        g = float3(hg, hg * velocity(q.x, q.y), hg * velocity(q.x, q.z));
    }
    return g;
}

public float3 ghost(uint side, float3 q, float z)
{
    return ghost_of(P.bc_kind[side], P.bc_value[side], P.bc_zmin[side], inflow_norm(side), AXIS[side], INWARD[side], q, z);
}
