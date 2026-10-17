// Per-frame scene parameters. Render pipelines use set 0 = solver set (Q, zb, P: see params.slang),
// set 1 = effect fields (effects_data.slang), set 2 = this uniform buffer. Mirror of src/render/SceneParams.h.
//
// Display coordinates (as in ui/renderer.py): physical (x, y, z) -> (x S, z z_scale S, -y S), Y up,
// S = 10 / max(Lx, Ly) so the domain is always ~10 units wide.
module scene;

import params;

public static const uint SCENE_LANDCOVER = 1;   // large domains: forests / parcels instead of plain grass
public static const uint SCENE_OUTSIDE   = 2;   // the `outside` mask is meaningful
public static const uint SCENE_ORTHO     = 4;   // an orthophoto covers the domain (sampled by terrain.slang)
public static const uint SCENE_THIN_WATER = 8;  // keep thin water films visible (minimum opacity)
public static const uint SCENE_ZONES     = 16;  // tint the cells of the solver zones (sources, sinks, levels)

public static const uint COLOR_REALISTIC = 0;
public static const uint COLOR_DEPTH     = 1;
public static const uint COLOR_SPEED     = 2;
public static const uint COLOR_FROUDE    = 3;
public static const uint COLOR_ELEVATION = 4;

public struct SceneParams           // std140, 176 bytes
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
    public float    cmap_min;       // colour map range (color_mode != COLOR_REALISTIC)
    public float    cmap_max;
    public float    cmap_opacity;
    public float    ortho_mix;      // 0 = procedural ground, 1 = orthophoto
    public uint4    sel;            // selected cell rectangle i0, j0, i1, j1 (inclusive); i1 < i0 = none
};

// cell of a display position (terrain / water fragments)
public int2 cell_of(float3 disp)
{
    return int2(int(floor(disp.x / (SC.S * P.dx))), int(floor(-disp.z / (SC.S * P.dy))));
}

public bool in_selection(int2 c)
{
    return SC.sel.z >= SC.sel.x && c.x >= int(SC.sel.x) && c.x <= int(SC.sel.z) && c.y >= int(SC.sel.y) && c.y <= int(SC.sel.w);
}

// zone tint of a cell (alpha 0 = none)
public float4 zone_tint(int2 c)
{
    if ((SC.flags & SCENE_ZONES) == 0 || c.x < 0 || c.y < 0 || c.x >= int(P.nx) || c.y >= int(P.ny))
        return float4(0.0f);
    uint z = cond[idx(uint(c.x), uint(c.y))];
    if (z == 0)
        return float4(0.0f);
    uint kind = zones[min(z - 1, MAX_ZONES - 1)].kind;
    if (kind == ZONE_SOURCE) return float4(0.25f, 0.55f, 1.0f, 0.45f);
    if (kind == ZONE_SINK)   return float4(1.0f, 0.3f, 0.2f, 0.45f);
    if (kind == ZONE_LEVEL)  return float4(0.2f, 0.95f, 0.9f, 0.45f);
    return float4(0.0f);
}

// selection + zone overlays applied to a shaded colour
public float3 overlays(float3 col, float3 disp)
{
    int2   c = cell_of(disp);
    float4 z = zone_tint(c);
    col = lerp(col, z.rgb, z.a);
    if (in_selection(c))
    {
        // orange tint with a 1-cell border
        bool border = c.x == int(SC.sel.x) || c.x == int(SC.sel.z) || c.y == int(SC.sel.y) || c.y == int(SC.sel.w);
        col = lerp(col, float3(1.0f, 0.6f, 0.15f), border ? 0.85f : 0.3f);
    }
    return col;
}

[[vk::binding(0, 2)]] public ConstantBuffer<SceneParams> SC;
// cells outside the useful domain (e.g. outside the catchment): greyed out (1 = outside)
[[vk::binding(1, 2)]] public StructuredBuffer<uint>      outside;
// orthophoto of the domain, row 0 = north edge; u = x / Lx, v = 1 - y / Ly (physical metres)
[[vk::binding(2, 2)]] public Sampler2D                   ortho;

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
