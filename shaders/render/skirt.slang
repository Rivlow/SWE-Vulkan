// Model base ("socle"): vertical walls along the 4 sides of the displayed rectangle (the domain, or the cut box),
// earth down to z_base and the water slice from the free surface down to the bed. Generated from SV_VertexID,
// 6 vertices per perimeter segment. Port of the skirt part of render() in ui/renderer.py.
// Groundwater cases: the earth wall is a block diagram of the soil column, coloured per pixel: unsaturated soil
// above the water table, saturated soil (blue) between the table and the substratum, bedrock below.
import params;
import scene;
import palette;
import noise;

struct SkirtVarying
{
    float4 pos   : SV_Position;
    float3 color : COLOR;
    float3 disp  : DISP;
};

struct SoilVarying
{
    float4 pos   : SV_Position;
    float  z     : ZPHYS;      // physical elevation of the pixel
    float  zb    : BED;        // ground, water table and substratum at this abscissa (linear along the wall)
    float  head  : GWHEAD;
    float  zsub  : ZSUB;
    float  shade : SHADE;
    float  along : ALONG;      // position along the wall (texture)
};

// p-th cell around the displayed rectangle (counter-clockwise) -> (i, j) and outward normal (display frame)
void perimeter(uint p, out uint i, out uint j, out float3 n)
{
    uint4 r = crop_rect();
    uint  w = r.z - r.x + 1, h = r.w - r.y + 1;
    if (p < w)                { i = r.x + p;               j = r.y;                     n = float3(0, 0, 1);  }  // south
    else if (p < w + h)       { i = r.z;                   j = r.y + p - w;             n = float3(1, 0, 0);  }  // east
    else if (p < 2 * w + h)   { i = r.z - (p - w - h);     j = r.w;                     n = float3(0, 0, -1); }  // north
    else                      { i = r.x;                   j = r.w - (p - 2 * w - h);   n = float3(-1, 0, 0); }  // west
}

// corner c of the quad of segment p: top/bottom of perimeter cells p and p+1
void corner(uint vid, out uint p, out bool top)
{
    uint4 r    = crop_rect();
    uint  seg  = vid / 6;
    uint  c    = vid % 6;                 // (p,top) (p,bot) (q,bot) (p,top) (q,bot) (q,top)
    uint  nper = 2 * ((r.z - r.x + 1) + (r.w - r.y + 1));
    bool  useQ = c == 2 || c == 4 || c == 5;
    p   = useQ ? (seg + 1) % nper : seg;
    top = c == 0 || c == 3 || c == 5;
}

[shader("vertex")]
SoilVarying skirtBedVert(uint vid : SV_VertexID)
{
    uint p; bool top;
    corner(vid, p, top);
    uint i, j; float3 n;
    perimeter(p, i, j, n);
    uint  k = idx(i, j);
    float z = zb[k];
    bool  gw_case = (P.flags & FLAG_GROUNDWATER) != 0;

    SoilVarying o;
    o.z     = top ? z : SC.z_base;
    o.pos   = project(display_pos(i, j, o.z));
    o.zb    = z;
    o.head  = gw_case ? min(gw_head(k), z) : SC.z_base - 1.0f;
    o.zsub  = gw_case ? soil[k].w : SC.z_base - 1.0f;
    o.shade = 0.55f + 0.45f * max(dot(n, SC.sun.xyz), 0.0f) + 0.1f;
    o.along = float(p) * min(P.dx, P.dy);
    return o;
}

[shader("fragment")]
float4 skirtBedFrag(SoilVarying v) : SV_Target
{
    float  t      = saturate((v.z - SC.z_base) / max(v.zb - SC.z_base, 1e-6f));
    float  stripe = 0.9f + 0.1f * sin(40.0f * v.z / SC.depth_ref);                 // faint strata
    float3 earth  = lerp(SOIL * 0.55f, SOIL * 1.15f * stripe, t) * v.shade;
    float3 col    = earth;
    if ((P.flags & FLAG_GROUNDWATER) != 0)
    {
        if (v.z <= v.zsub)
        {
            // bedrock: grey, coarse grain
            float g = 0.85f + 0.15f * value_noise(3.0f * v.along / SC.depth_ref, 8.0f * v.z / SC.depth_ref);
            col = STONE * 0.75f * g * v.shade;
        }
        else if (v.z <= v.head)
        {
            // saturated soil: the earth soaked in blue, with short horizontal dashes (water in the pores)
            float dash = frac(6.0f * v.z / SC.depth_ref) < 0.25f && frac(4.0f * v.along / SC.depth_ref) < 0.5f ? 0.25f : 0.0f;
            col = lerp(earth, float3(0.15f, 0.35f, 0.8f) * v.shade, 0.55f + dash);
        }
        // the water table itself: a bright line of constant width on screen
        float d = abs(v.z - v.head) / max(fwidth(v.z), 1e-6f);
        col = lerp(col, float3(0.4f, 0.8f, 1.0f), 1.0f - sstep(0.8f, 2.0f, d));
    }
    return float4(col, 1.0f);
}

// ---- raw cells (SCENE_BLOCKS): one flat-topped wall per perimeter cell, from corner to corner of its outer edge,
// so the steps between neighbouring cells come for free (each wall is a full rectangle down to z_base)
void block_edge(uint vid, out uint p, out uint i, out uint j, out float2 corner, out bool top, out float3 n)
{
    uint seg = vid / 6, c = vid % 6;        // (A,top) (A,bot) (B,bot) (A,top) (B,bot) (B,top)
    uint4 r    = crop_rect();
    uint  nper = 2 * ((r.z - r.x + 1) + (r.w - r.y + 1));
    p = seg % nper;
    perimeter(p, i, j, n);
    bool useB = c == 2 || c == 4 || c == 5;
    top       = c == 0 || c == 3 || c == 5;
    float2 a, b;
    if (n.z > 0.5f)       { a = float2(i, j);         b = float2(i + 1, j); }          // south edge
    else if (n.x > 0.5f)  { a = float2(i + 1, j);     b = float2(i + 1, j + 1); }      // east
    else if (n.z < -0.5f) { a = float2(i + 1, j + 1); b = float2(i, j + 1); }          // north
    else                  { a = float2(i, j + 1);     b = float2(i, j); }              // west
    corner = useB ? b : a;
}

[shader("vertex")]
SoilVarying skirtBlockBedVert(uint vid : SV_VertexID)
{
    uint p, i, j; float2 c; bool top; float3 n;
    block_edge(vid, p, i, j, c, top, n);
    uint  k = idx(i, j);
    float z = zb[k];
    bool  gw_case = (P.flags & FLAG_GROUNDWATER) != 0;

    SoilVarying o;
    o.z     = top ? z : SC.z_base;
    o.pos   = project(corner_pos(c.x, c.y, o.z));
    o.zb    = z;
    o.head  = gw_case ? min(gw_head(k), z) : SC.z_base - 1.0f;
    o.zsub  = gw_case ? soil[k].w : SC.z_base - 1.0f;
    o.shade = 0.55f + 0.45f * max(dot(n, SC.sun.xyz), 0.0f) + 0.1f;
    o.along = (float(p) + (vid % 6 == 2 || vid % 6 == 4 || vid % 6 == 5 ? 1.0f : 0.0f)) * min(P.dx, P.dy);
    return o;
}

[shader("vertex")]
SkirtVarying skirtBlockWaterVert(uint vid : SV_VertexID)
{
    uint p, i, j; float2 c; bool top; float3 n;
    block_edge(vid, p, i, j, c, top, n);
    uint  k     = idx(i, j);
    float z     = zb[k];
    float h     = Q[k].x;
    float eta   = z + (h > 1e-3f ? h : 0.0f);
    float shade = 0.55f + 0.45f * max(dot(n, SC.sun.xyz), 0.0f) + 0.1f;

    SkirtVarying o;
    o.disp  = corner_pos(c.x, c.y, top ? eta : z);
    o.pos   = project(o.disp);
    o.color = top ? lerp(SHALLOW, DEEP, 0.4f) * shade : DEEP * shade;
    return o;
}

[shader("vertex")]
SkirtVarying skirtWaterVert(uint vid : SV_VertexID)
{
    uint p; bool top;
    corner(vid, p, top);
    uint i, j; float3 n;
    perimeter(p, i, j, n);
    uint  k     = idx(i, j);
    float z     = zb[k];
    float h     = Q[k].x;
    float eta   = z + (h > 1e-3f ? h : 0.0f);
    float shade = 0.55f + 0.45f * max(dot(n, SC.sun.xyz), 0.0f) + 0.1f;

    SkirtVarying o;
    o.disp  = display_pos(i, j, top ? eta : z);
    o.pos   = project(o.disp);
    o.color = top ? lerp(SHALLOW, DEEP, 0.4f) * shade : DEEP * shade;
    return o;
}

[shader("fragment")]
float4 skirtFrag(SkirtVarying v) : SV_Target
{
    return float4(v.color, 1.0f);
}
