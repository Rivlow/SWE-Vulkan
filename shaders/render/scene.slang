// Per-frame scene parameters. Render pipelines use set 0 = solver set (Q, zb, P: see params.slang),
// set 1 = effect fields (effects_data.slang), set 2 = this uniform buffer. Mirror of src/render/SceneParams.h.
//
// Display coordinates (as in ui/renderer.py): physical (x, y, z) -> (x S, z z_scale S, -y S), Y up,
// S = 10 / max(Lx, Ly) so the domain is always ~10 units wide.
module scene;

import params;

public static const uint SCENE_LANDCOVER = 1;   // large domains: forests / parcels instead of plain grass
public static const uint SCENE_OUTSIDE   = 2;   // the `outside` mask is meaningful

public struct SceneParams           // std140, 144 bytes
{
    public float4x4 view_proj;
    public float4   cam_pos;        // xyz
    public float4   sun;            // xyz, normalised
    public float    S;
    public float    z_scale;
    public float    depth_ref;
    public float    water_clarity;
    public float    z_base;         // bottom of the model base
    public float    time;
    public float    Lx;
    public float    Ly;
    public uint     flags;
    public uint     color_mode;
    public float    viewport_w;     // pixels (screen-space line width)
    public float    viewport_h;
};

[[vk::binding(0, 2)]] public ConstantBuffer<SceneParams> SC;
// cells outside the useful domain (e.g. outside the catchment): greyed out (1 = outside)
[[vk::binding(1, 2)]] public StructuredBuffer<uint>      outside;

public float outside_at(uint k)
{
    return (SC.flags & SCENE_OUTSIDE) != 0 && outside[k] != 0 ? 1.0f : 0.0f;
}

public int2 clamp_ij(int i, int j)
{
    return int2(clamp(i, 0, int(P.nx) - 1), clamp(j, 0, int(P.ny) - 1));
}

public float bed_height(int i, int j)
{
    int2 c = clamp_ij(i, j);
    return zb[idx(c.x, c.y)];
}

public float depth_at(int i, int j)
{
    int2 c = clamp_ij(i, j);
    return Q[idx(c.x, c.y)].x;
}

public float water_height(int i, int j)
{
    int2 c = clamp_ij(i, j);
    uint k = idx(c.x, c.y);
    return zb[k] + Q[k].x;
}

// Display position of the cell centre (i, j) at physical elevation z
public float3 display_pos(uint i, uint j, float z)
{
    return float3((float(i) + 0.5f) * P.dx * SC.S, z * SC.z_scale * SC.S, -(float(j) + 0.5f) * P.dy * SC.S);
}

public float4 project(float3 p)
{
    return mul(SC.view_proj, float4(p, 1.0f));
}
