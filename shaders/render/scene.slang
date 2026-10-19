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
public static const uint SCENE_SELECTION = 256; // the selection mask (binding 3) is meaningful
public static const uint SCENE_BORDERS   = 512; // draw the boundary conditions (open sides, face conditions)
public static const uint SCENE_MESH      = 1024; // draw the cell boundaries ("Maillage")
public static const uint SCENE_BLOCKS    = 2048; // raw cells: flat-topped columns, no interpolation (block vertex shaders)

// highlight codes (SC.sel.w): a side of the domain or a group of face conditions, e.g. hovered in the GUI
public static const uint HIGHLIGHT_SIDE  = 1;   // + side
public static const uint HIGHLIGHT_GROUP = 16;  // + group

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
    public uint4    sel;            // hovered cell i, j (~0 = none), unused, highlight code
    public float4   gw;             // x = contour step of the water table (m), y = contour opacity, z = focal length (px)
    public uint4    crop;           // cut box i0, j0, i1, j1 (inclusive, SCENE_CROP)
};

// Displayed cell rectangle: the cut box, or the whole domain
public uint4 crop_rect()
{
    return (SC.flags & SCENE_CROP) != 0 ? SC.crop : uint4(0, 0, P.nx - 1, P.ny - 1);
}

// Fragment outside the cut box (the box ends at the centres of its edge cells, like the meshes and walls; in the
// raw-cell mode the cells are whole, so the box ends at their outer edges)
public bool cropped_out(float3 disp)
{
    if ((SC.flags & SCENE_CROP) == 0)
        return false;
    float x = disp.x / (SC.S * P.dx) - 0.5f, y = -disp.z / (SC.S * P.dy) - 0.5f;
    float e = (SC.flags & SCENE_BLOCKS) != 0 ? 0.5f + 1e-3f : 1e-3f;
    return x < float(SC.crop.x) - e || x > float(SC.crop.z) + e || y < float(SC.crop.y) - e || y > float(SC.crop.w) + e;
}

// ---------------------------------------------------------------------------------------------
// Raw cells (SCENE_BLOCKS): every cell is a flat-topped column. 18 vertices per cell, no index buffer: the top
// quad (part 0), the east wall (part 1, shared with cell i+1) and the north wall (part 2, shared with cell j+1),
// 6 vertices each. A wall between two columns of the same height, or on the domain's edge (the block skirt
// draws those), is collapsed to a line (no fragments). The wall belongs to the taller column: its overlays
// (selection, zones, face conditions) are those of that cell and of the face it exposes.
// ---------------------------------------------------------------------------------------------
static const float2 QUAD[6] = { float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 0), float2(1, 1), float2(0, 1) };

// Display position of the cell corner (i, j) (cell units, corners at integers) at physical elevation z
public float3 corner_pos(float i, float j, float z)
{
    return float3(i * P.dx * SC.S, z * SC.z_scale * SC.S, -j * P.dy * SC.S);
}

public struct BlockVertex
{
    public float3 disp;
    public float3 n;
    public uint   owner;      // cell whose overlays apply
    public uint   side;       // walls: face of the owner (0 west, 1 east, 2 south, 3 north); tops: 0
    public bool   wall;
};

// Vertex `vid` of the block geometry, given the column tops: z0 of the cell, ze / zn of its east / north
// neighbours (has_east / has_north false on the domain's edge), and `flat` true to collapse the top (dry water)
public BlockVertex block_vertex(uint vid, uint cell, float z0, float ze, bool has_east, float zn, bool has_north, bool flat)
{
    uint   part = (vid % 18) / 6, k = vid % 6;
    uint   i = cell / P.ny, j = cell % P.ny;
    float2 q = QUAD[k];
    BlockVertex o;
    o.owner = cell;
    o.side  = 0;
    o.wall  = part != 0;
    if (part == 0)
    {
        o.n    = float3(0.0f, 1.0f, 0.0f);
        o.disp = flat ? corner_pos(float(i), float(j), z0) : corner_pos(float(i) + q.x, float(j) + q.y, z0);
        return o;
    }
    bool  east = part == 1;
    bool  has  = east ? has_east : has_north;
    float zn_  = east ? ze : zn;
    bool  collapse = !has || zn_ == z0;
    float z    = collapse ? z0 : (q.y == 0.0f ? z0 : zn_);
    // the taller column owns the wall; the face it exposes looks towards the lower one
    bool  mine = collapse || z0 >= zn_;
    o.owner = mine ? cell : (east ? cell + P.ny : cell + 1);
    o.side  = east ? (mine ? 1u : 0u) : (mine ? 3u : 2u);
    o.n     = east ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 0.0f, -1.0f);
    o.disp  = east ? corner_pos(float(i) + 1.0f, float(j) + (collapse ? 0.0f : q.x), z)
                   : corner_pos(float(i) + (collapse ? 0.0f : q.x), float(j) + 1.0f, z);
    return o;
}

// Overlays of a block wall: the owner's zone, selection and hover tints, and the condition of the exposed face
// (face conditions, or the open sides of the domain) over the whole face
public float3 wall_overlays(float3 col, uint owner, uint side)
{
    int2   c = int2(int(owner / P.ny), int(owner % P.ny));
    float4 z = zone_tint(c);
    col = lerp(col, z.rgb, z.a);
    if ((SC.flags & SCENE_BORDERS) != 0)
    {
        uint b    = (face_flags[owner] >> (8 * side)) & 0xffu;
        bool edge = (side == 0 && c.x == 0) || (side == 1 && c.x + 1 == int(P.nx)) || (side == 2 && c.y == 0) || (side == 3 && c.y + 1 == int(P.ny));
        uint kind = 0;
        bool hl = false, on = false;
        if ((b & 7u) != 0)
        {
            kind = (b & 7u) - 1u;
            hl   = SC.sel.w == HIGHLIGHT_GROUP + (b >> 3);
            on   = true;
        }
        else if (edge)
        {
            kind = P.bc_kind[side];
            hl   = SC.sel.w == HIGHLIGHT_SIDE + side;
            on   = kind != BC_WALL || hl;
        }
        if (on)
        {
            float3 tint = bc_color(kind);
            col = lerp(col, hl ? lerp(tint, float3(1.0f), 0.5f) : tint, 0.85f);
        }
    }
    if (in_selection(c))
        col = lerp(col, SELECTION, 0.32f);
    if (is_hovered(c))
        col = lerp(col, float3(1.0f), 0.18f);
    return col;
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

bool in_grid(int2 c)
{
    return c.x >= 0 && c.y >= 0 && c.x < int(P.nx) && c.y < int(P.ny);
}

// Cell of the selection mask (one bit per cell)
public bool in_selection(int2 c)
{
    if ((SC.flags & SCENE_SELECTION) == 0 || !in_grid(c))
        return false;
    uint k = idx(uint(c.x), uint(c.y));
    return ((selmask[k >> 5] >> (k & 31u)) & 1u) != 0;
}

public bool is_hovered(int2 c)
{
    return SC.sel.x != ~0u && c.x == int(SC.sel.x) && c.y == int(SC.sel.y);
}

// Colour of a boundary condition kind (BC_*), same as src/render/Colormap.cpp bcColor()
public float3 bc_color(uint kind)
{
    switch (kind)
    {
    case BC_INFLOW: return float3(0.25f, 0.55f, 1.00f);
    case BC_LEVEL:  return float3(0.10f, 0.85f, 0.75f);
    case BC_FREE:   return float3(0.72f, 0.48f, 1.00f);
    case BC_DEPTH:  return float3(0.98f, 0.40f, 0.78f);
    default:        return float3(0.55f, 0.57f, 0.62f);
    }
}

// Boundary conditions of cell c: for each side, a band along the face coloured by the kind. Face conditions
// (face_flags: per side a byte kind + 1 | group << 3) and the open sides of the domain; a highlighted side or
// group is drawn wider and lighter, walls of the domain only when highlighted.
float3 border_overlay(float3 col, int2 c, float2 f, float2 px)
{
    if ((SC.flags & SCENE_BORDERS) == 0 || !in_grid(c))
        return col;
    uint   flags = face_flags[idx(uint(c.x), uint(c.y))];
    float2 wb    = clamp(2.5f * px, 0.06f, 0.5f);        // ~2.5 px on screen
    for (uint side = 0; side < 4; ++side)
    {
        uint b    = (flags >> (8 * side)) & 0xffu;
        bool edge = (side == 0 && c.x == 0) || (side == 1 && c.x + 1 == int(P.nx)) ||
                    (side == 2 && c.y == 0) || (side == 3 && c.y + 1 == int(P.ny));
        uint kind;
        bool hl;
        if ((b & 7u) != 0)
        {
            kind = (b & 7u) - 1u;
            hl   = SC.sel.w == HIGHLIGHT_GROUP + (b >> 3);
        }
        else
        {
            if (!edge)
                continue;
            kind = P.bc_kind[side];
            hl   = SC.sel.w == HIGHLIGHT_SIDE + side;
            if (kind == BC_WALL && !hl)
                continue;
        }
        float d = side == 0 ? f.x : side == 1 ? 1.0f - f.x : side == 2 ? f.y : 1.0f - f.y;
        // the meshes join the cell centres: the outer half of an edge cell is not drawn, so a band on a face of
        // the domain's edge sits against the drawn edge (the cell centre)
        if (edge)
            d -= 0.5f;
        float w = (side < 2 ? wb.x : wb.y) * (hl ? 1.8f : 1.0f);
        if (d < w)
        {
            float3 tint = bc_color(kind);
            col = lerp(col, hl ? lerp(tint, float3(1.0f), 0.5f) : tint, 0.95f);
        }
    }
    return col;
}

// zone tint of a cell (alpha 0 = none)
public float4 zone_tint(int2 c)
{
    if ((SC.flags & SCENE_ZONES) == 0 || c.x < 0 || c.y < 0 || c.x >= int(P.nx) || c.y >= int(P.ny))
        return float4(0.0f);
    if (bridge_at(idx(uint(c.x), uint(c.y))))
        return float4(0.95f, 0.6f, 0.25f, 0.5f);        // under a bridge deck (Conditions > Ouvrages)
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

public static const float3 SELECTION = float3(1.0f, 0.80f, 0.20f);   // src/render/SceneParams.h kSelectionColor

// Cell boundaries ("Maillage"): lines of constant width on screen at the edges of the cells (the render mesh
// joins the cell centres, so its own edges would sit half a cell off). The lines fade with the size of a cell
// on screen: far away (cells of a few pixels) they would turn the whole terrain black.
float3 mesh_overlay(float3 col, float2 f, float2 px)
{
    if ((SC.flags & SCENE_MESH) == 0)
        return col;
    float cellPx = 1.0f / max(max(px.x, px.y), 1e-6f);
    float fade   = 0.85f * smoothstep(3.0f, 14.0f, cellPx);
    if (fade < 0.01f)
        return col;
    float2 d    = min(f, 1.0f - f) / max(px, 1e-6f);        // distance to the nearest edge, in pixels
    float  line = 1.0f - smoothstep(0.5f, 1.5f, min(d.x, d.y));
    return lerp(col, float3(0.08f, 0.10f, 0.12f), line * fade);
}

// Mesh, selection, hovered cell, boundary conditions and zone overlays applied to a shaded colour
public float3 overlays(float3 col, float3 disp)
{
    int2   c  = cell_of(disp);
    float2 u  = float2(disp.x / (SC.S * P.dx), -disp.z / (SC.S * P.dy));        // position in cell units
    float2 f  = u - float2(c);                                                   // position in the cell
    float2 px = fwidth(u);        // one pixel, in cell units (from the unwrapped coordinate: f jumps at the edges)
    float4 z  = zone_tint(c);
    col = lerp(col, z.rgb, z.a);
    col = chrono_overlay(col, disp, c);
    col = mesh_overlay(col, f, px);
    col = border_overlay(col, c, f, px);
    if (in_selection(c))
    {
        // every selected cell is tinted; the outline of the selected set (faces towards unselected cells) is a
        // crisp line of constant width on screen
        float2 w    = clamp(1.6f * px, 0.03f, 0.45f);
        bool   edge = (f.x < w.x && !in_selection(c - int2(1, 0))) || (f.x > 1.0f - w.x && !in_selection(c + int2(1, 0))) ||
                      (f.y < w.y && !in_selection(c - int2(0, 1))) || (f.y > 1.0f - w.y && !in_selection(c + int2(0, 1)));
        col = lerp(col, SELECTION, edge ? 0.95f : 0.32f);
    }
    if (is_hovered(c))
    {
        float2 w    = clamp(1.3f * px, 0.03f, 0.45f);
        bool   edge = f.x < w.x || f.x > 1.0f - w.x || f.y < w.y || f.y > 1.0f - w.y;
        col = lerp(col, float3(1.0f), edge ? 0.9f : 0.18f);
    }
    return col;
}

[[vk::binding(0, 2)]] public ConstantBuffer<SceneParams> SC;
// cells outside the useful domain (e.g. outside the catchment): greyed out (1 = outside)
[[vk::binding(1, 2)]] public StructuredBuffer<uint>      outside;
// orthophoto of the domain, row 0 = north edge; u = x / Lx, v = 1 - y / Ly (physical metres)
[[vk::binding(2, 2)]] public Sampler2D                   ortho;
// selected cells, one bit per cell (bit k & 31 of word k >> 5), meaningful with SCENE_SELECTION
[[vk::binding(3, 2)]] public StructuredBuffer<uint>      selmask;
// face conditions per cell: one byte per side (west, east, south, north): 0 none, else kind + 1 | group << 3
[[vk::binding(4, 2)]] public StructuredBuffer<uint>      face_flags;

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
