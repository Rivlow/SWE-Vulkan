// Per-frame scene parameters. Render pipelines use set 0 = solver set (Q, zb, P: see params.slang),
// set 1 = effect fields (effects_data.slang), set 2 = this uniform buffer. Mirror of src/render/SceneParams.h.
//
// Display coordinates (as in ui/renderer.py): physical (x, y, z) -> (x S, z z_scale S, -y S), Y up,
// S = 10 / max(Lx, Ly) so the domain is always ~10 units wide.
module scene;

import params;
import groundwater;

public static const uint SCENE_LANDCOVER = 1;   // large domains: forests / parcels instead of plain grass
public static const uint SCENE_OUTSIDE   = 2;   // the `outside` mask is meaningful
public static const uint SCENE_ORTHO     = 4;   // an orthophoto covers the domain (sampled by terrain.slang)
public static const uint SCENE_THIN_WATER = 8;  // keep thin water films visible (minimum opacity)
public static const uint SCENE_ZONES     = 16;  // tint the cells of the solver zones (sources, sinks, levels)
public static const uint SCENE_GW_CONTOURS = 32; // contour lines of the water table on the ground (step SC.gw.x)

public static const uint COLOR_REALISTIC = 0;
public static const uint COLOR_DEPTH     = 1;
public static const uint COLOR_SPEED     = 2;
public static const uint COLOR_FROUDE    = 3;
public static const uint COLOR_ELEVATION = 4;
// groundwater quantities: shown on the ground (terrain mesh), the water keeps its realistic look
public static const uint COLOR_GW_TABLE_DEPTH = 5;   // depth of the water table below the ground zb - H (m)
public static const uint COLOR_GW_SATURATION  = 6;   // saturated fraction of the soil h_gw / (zb - z_sub)
public static const uint COLOR_GW_EXCHANGE    = 7;   // exchange rate (mm/h, > 0 = infiltration), diverging map
public static const uint COLOR_GW_INFILTRATED = 8;   // net infiltrated depth since t = 0 (mm)
public static const uint COLOR_GW_HEAD        = 9;   // water table elevation H = z_sub + h_gw (m)
public static const uint COLOR_RAIN           = 10;  // rain depth since t = 0 (mm), chronology cases

public static const uint SCENE_CROP      = 64;  // cut box: only the cells crop.xy .. crop.zw are drawn, the walls follow it
public static const uint SCENE_GW_XRAY   = 128; // water table surface seen through the ground

public struct SceneParams           // std140, 208 bytes
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
    public float4   gw;             // x = contour step of the water table (m), y = contour opacity, z = focal length (px)
    public uint4    crop;           // cut box i0, j0, i1, j1 (inclusive, SCENE_CROP)
};

// Wireframe overlays (terrain and water)
public struct WireVarying
{
    public float4 pos  : SV_Position;
    public float3 disp : DISP;
};

// Displayed cell rectangle: the cut box, or the whole domain
public uint4 crop_rect()
{
    return (SC.flags & SCENE_CROP) != 0 ? SC.crop : uint4(0, 0, P.nx - 1, P.ny - 1);
}

// Fragment outside the cut box (the box ends at the centres of its edge cells, like the meshes and walls)
public bool cropped_out(float3 disp)
{
    if ((SC.flags & SCENE_CROP) == 0)
        return false;
    float x = disp.x / (SC.S * P.dx) - 0.5f, y = -disp.z / (SC.S * P.dy) - 0.5f;
    float e = 1e-3f;
    return x < float(SC.crop.x) - e || x > float(SC.crop.z) + e || y < float(SC.crop.y) - e || y > float(SC.crop.w) + e;
}

// Ground colouring on: a groundwater quantity is chosen and the case has an aquifer, or the cumulated rain and
// the case has chronologies
public bool gw_color_mode()
{
    if (SC.color_mode == COLOR_RAIN)
        return (P.flags & FLAG_SOURCES) != 0;
    return SC.color_mode >= COLOR_GW_TABLE_DEPTH && (P.flags & FLAG_GROUNDWATER) != 0;
}

// Water table elevation of cell k (relative to z_ref)
public float gw_head(uint k)
{
    return soil[k].w + gw[k].x;
}

// Groundwater quantity of the colour map at cell k
public float gw_color_value(uint k)
{
    if (SC.color_mode == COLOR_RAIN)
        return source_cumul(k).y * 1e3f;
    float4 s = soil[k], g = gw[k];
    switch (SC.color_mode)
    {
    case COLOR_GW_TABLE_DEPTH: return zb[k] - (s.w + g.x);
    case COLOR_GW_SATURATION:  return g.x / max(zb[k] - s.w, 1e-3f);
    case COLOR_GW_EXCHANGE:    return g.y * 3.6e6f;
    case COLOR_GW_INFILTRATED: return (g.z - g.w) * 1e3f;
    case COLOR_GW_HEAD:        return s.w + g.x;
    default:                   return 0.0f;
    }
}

// Diverging map for signed quantities: blue (t = 0, water leaving the ground), white (0.5), brown (t = 1,
// water entering the ground). Same colours in src/render/Colormap.cpp (legend).
public float3 diverging(float t)
{
    float3 blue = float3(0.12f, 0.38f, 0.85f), white = float3(0.95f, 0.95f, 0.92f), brown = float3(0.55f, 0.32f, 0.10f);
    t = saturate(t);
    return t < 0.5f ? lerp(blue, white, 2.0f * t) : lerp(white, brown, 2.0f * t - 1.0f);
}

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

// Chronology zone (part = 0: infiltration, 1: rain) of cell c, 0 = none or outside
uint chrono_zone(int2 c, uint part)
{
    if (c.x < 0 || c.y < 0 || c.x >= int(P.nx) || c.y >= int(P.ny))
        return 0;
    return (src_zones[idx(uint(c.x), uint(c.y))] >> (16 * part)) & 0xffffu;
}

// Rain (blue, one diagonal) and infiltration (brown, the other diagonal) zones: hatching, and a border where the
// zone changes so that neighbouring zones stay apart
float3 chrono_overlay(float3 col, float3 disp, int2 c)
{
    if ((SC.flags & SCENE_ZONES) == 0 || (P.flags & FLAG_SOURCES) == 0)
        return col;
    float2 f = float2(disp.x / (SC.S * P.dx), -disp.z / (SC.S * P.dy)) - float2(c);        // position in the cell
    for (uint part = 0; part < 2; ++part)
    {
        uint z = chrono_zone(c, part);
        if (z == 0)
            continue;
        float3 tint  = part == 1 ? float3(0.2f, 0.45f, 1.0f) : float3(0.6f, 0.35f, 0.1f);
        float  s     = part == 1 ? disp.x + disp.z : disp.x - disp.z;
        // fine hatching at a constant pitch on screen (~14 px), readable whatever the size of the zone
        float  pitch = max(fwidth(s), 1e-6f) * 14.0f;
        float  hatch = frac(s / pitch) < 0.22f ? 0.22f : 0.0f;
        bool   edge  = (f.x < 0.12f && chrono_zone(c - int2(1, 0), part) != z) || (f.x > 0.88f && chrono_zone(c + int2(1, 0), part) != z) ||
                       (f.y < 0.12f && chrono_zone(c - int2(0, 1), part) != z) || (f.y > 0.88f && chrono_zone(c + int2(0, 1), part) != z);
        col = lerp(col, tint, edge ? 0.85f : hatch);
    }
    return col;
}

// selection + zone overlays applied to a shaded colour
public float3 overlays(float3 col, float3 disp)
{
    int2   c = cell_of(disp);
    float4 z = zone_tint(c);
    col = lerp(col, z.rgb, z.a);
    col = chrono_overlay(col, disp, c);
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
