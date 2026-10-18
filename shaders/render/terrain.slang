// Bed mesh: one vertex per cell centre, fetched from the solver buffers (no vertex input).
// Port of the bed part of render() / terrain_color() in ui/renderer.py, shaded per pixel.
import params;
import effects_data;
import scene;
import noise;
import palette;
import colormap;

struct TerrainVarying
{
    float4 pos      : SV_Position;
    float3 disp     : DISP;        // display position (procedural textures use display x, z like the Python)
    float3 n        : NORMAL;      // with vertical exaggeration
    float  slope    : SLOPE;       // real slope (no exaggeration)
    float  z        : BED;
    float  wet      : WET;         // 1 where h > 2e-3
    float  wet_blur : WETBLUR;     // blurred wet mask -> sand band
    float  out      : OUTSIDE;     // 1 outside the useful domain
    float  flooded  : FLOODED;     // reached by the water at some point
    float  wet_mem  : WETMEM;      // wet ground drying slowly
    float  shadow   : SHADOW;      // sun visibility (cast shadows)
    float  ao       : AO;
    float  gval     : GWVALUE;     // groundwater quantity of the colour map
    float  head     : GWHEAD;      // water table elevation (contour lines)
};

[shader("vertex")]
TerrainVarying terrainVert(uint vid : SV_VertexID)
{
    int i = int(vid / P.ny), j = int(vid % P.ny);
    float z = zb[vid];

    float dzx = (bed_height(i + 1, j) - bed_height(i - 1, j)) / (2.0f * P.dx) * SC.z_scale;
    float dzy = (bed_height(i, j + 1) - bed_height(i, j - 1)) / (2.0f * P.dy) * SC.z_scale;

    TerrainVarying o;
    o.disp     = display_pos(i, j, z);
    o.pos      = project(o.disp);
    o.n        = float3(-dzx, 1.0f, dzy);      // Z = -y
    o.slope    = sqrt(dzx * dzx + dzy * dzy) / SC.z_scale;
    o.z        = z;
    o.wet      = wet[vid];
    o.wet_blur = wet_blur[vid];
    o.out      = outside_at(vid);
    o.flooded  = flooded[vid];
    o.wet_mem  = wet_mem[vid];
    o.shadow   = shadow[vid];
    o.ao       = ao[vid];
    bool gw_on = (P.flags & FLAG_GROUNDWATER) != 0;
    o.gval     = gw_color_mode() ? gw_color_value(vid) : 0.0f;
    o.head     = gw_on && (SC.flags & SCENE_GW_CONTOURS) != 0 ? gw_head(vid) : 0.0f;
    return o;
}

// Value noise projected on the three axis planes and blended by the normal: no stretching on cliffs
// (on a near-horizontal face it reduces to the Python value_noise(f x, f y)).
float triplanar_noise(float3 p, float3 n, float f)
{
    float3 w = abs(n);
    w = pow(w, 4.0f);
    w /= w.x + w.y + w.z;
    return w.y * value_noise(f * p.x, f * p.z) + w.x * value_noise(f * p.z, f * p.y) + w.z * value_noise(f * p.x, f * p.y);
}

// Plausible land cover (no data): forest on steep slopes and in large patches, elsewhere a mosaic of
// slanted parcels (meadows, cereals, ploughed fields)
float3 land_color(float x, float y, float slope)
{
    float  forest = sstep(0.4f, 0.55f, 0.9f * sstep(0.05f, 0.2f, slope) + 0.5f * fbm(1.3f * x + 7.0f, 1.3f * y));
    float3 canopy = FOREST * (0.8f + 0.4f * value_noise(25.0f * x, 25.0f * y));
    float  c = 0.9f, s = 0.43f;                                     // ~25 deg rotation
    float  px = (c * x - s * y) * 6.0f, py = (s * x + c * y) * 9.0f;
    float  cell = hash2(floor(px), floor(py));
    float3 meadow = MEADOW * (0.88f + 0.24f * hash2(floor(px) + 3.1f, floor(py)));
    float3 parcel = meadow;
    if (cell > 0.88f)
        parcel = lerp(meadow, CROP_A, 0.6f);
    else if (cell > 0.8f)
        parcel = lerp(meadow, CROP_B, 0.5f);
    return lerp(parcel, canopy, forest);
}

float3 terrain_color(TerrainVarying v, float3 n)
{
    float x = v.disp.x, y = v.disp.z;
    float3 grass;
    if ((SC.flags & SCENE_LANDCOVER) != 0)
        // outside the domain: plain tint (no useless detail)
        grass = v.out > 0.5f ? lerp(GRASS_A, GRASS_B, fbm(1.6f * x, 1.6f * y)) : land_color(x, y, v.slope);
    else
    {
        float t = fbm(1.6f * x, 1.6f * y);
        grass = lerp(GRASS_A, GRASS_B, sstep(0.35f, 0.65f, t));
        grass *= 0.85f + 0.3f * value_noise(9.0f * x, 9.0f * y);
    }
    // sand around the water (noisy boundary), wet under the water
    float  shore = sstep(0.02f, 0.25f, v.wet_blur + 0.12f * (fbm(3.0f * x, 3.0f * y) - 0.5f));
    float3 sand  = lerp(SAND, WET_SAND, v.wet);
    sand *= 0.92f + 0.16f * value_noise(14.0f * x, 14.0f * y);
    float3 col = lerp(grass, sand, shore);
    // stone on steep slopes (quays, banks), lighter grass higher up
    float3 stone = STONE * (0.85f + 0.3f * triplanar_noise(v.disp, n, 6.0f));
    col = lerp(col, stone, sstep(0.75f, 0.45f, n.y));
    col *= 1.0f + 0.15f * min(v.z / SC.depth_ref, 1.0f);
    // orthophoto drape (geographic imports): real ground cover instead of the procedural one
    if ((SC.flags & SCENE_ORTHO) != 0 && SC.ortho_mix > 0.0f)
    {
        float2 uv  = float2(x / (SC.S * SC.Lx), 1.0f + y / (SC.S * SC.Ly));     // y = -y_phys * S
        float3 pic = ortho.Sample(uv).rgb;
        pic = pow(pic, 1.1f) * 1.05f;                                            // a touch of contrast
        // keep the wet sand under / near the water so the shoreline still reads
        float3 wetted = lerp(pic, pic * lerp(1.0f, 0.75f, v.wet), shore);
        col = lerp(col, wetted, SC.ortho_mix);
    }
    // after the water has passed: darker ground drying slowly, and a light silt deposit over the whole
    // flooded footprint (flood mark)
    float dry = 1.0f - v.wet;
    col = lerp(col, MUD_DRY, 0.3f * v.flooded * dry);
    col *= 1.0f - 0.35f * v.wet_mem * dry;
    // outside the useful domain: desaturated and darkened
    float  grey = dot(col, float3(0.3f, 0.5f, 0.2f));
    float3 dim  = 0.6f * lerp(col, float3(grey), 0.8f);
    return lerp(col, dim, v.out);
}

[shader("fragment")]
float4 terrainFrag(TerrainVarying v) : SV_Target
{
    if (cropped_out(v.disp))
        discard;
    float3 n    = normalize(v.n);
    float3 base = terrain_color(v, n);
    // groundwater map on the ground (the water above keeps its realistic look)
    if (gw_color_mode())
    {
        float  t   = (v.gval - SC.cmap_min) / max(SC.cmap_max - SC.cmap_min, 1e-12f);
        float3 map = SC.color_mode == COLOR_GW_EXCHANGE ? diverging(t) : turbo(t);
        base = lerp(base, map, SC.cmap_opacity);
    }
    // contour lines of the water table, constant width on screen
    if ((SC.flags & SCENE_GW_CONTOURS) != 0 && (P.flags & FLAG_GROUNDWATER) != 0 && SC.gw.x > 0.0f)
    {
        float x    = v.head / SC.gw.x;
        float d    = abs(frac(x + 0.5f) - 0.5f) / max(fwidth(x), 1e-6f);        // distance to a line, in pixels
        float line = 1.0f - sstep(0.6f, 1.6f, d);
        bool  major = abs(round(x) - 5.0f * round(x / 5.0f)) < 0.5f;             // every 5th line darker
        base = lerp(base, float3(0.05f, 0.18f, 0.45f), line * SC.gw.y * (major ? 1.0f : 0.6f));
    }
    return float4(overlays(light(base, n, SC.sun.xyz, 0.55f, 0.75f, v.ao, v.shadow), v.disp), 1.0f);
}

// Wireframe overlay ("Maillage"): position only
[shader("vertex")]
WireVarying terrainWireVert(uint vid : SV_VertexID)
{
    WireVarying o;
    o.disp = display_pos(vid / P.ny, vid % P.ny, zb[vid]);
    o.pos  = project(o.disp);
    return o;
}

[shader("fragment")]
float4 wireFrag(WireVarying v) : SV_Target
{
    if (cropped_out(v.disp))
        discard;
    // the lines fade with the size of a cell on screen: far away (cells of a few pixels) the mesh would turn the
    // whole terrain black. SC.gw.z = focal length in pixels; cell size on screen = focal * cell / distance.
    float dist   = max(length(v.disp - SC.cam_pos.xyz), 1e-3f);
    float cellPx = SC.gw.z * min(P.dx, P.dy) * SC.S / dist;
    float alpha  = 0.85f * smoothstep(3.0f, 14.0f, cellPx);
    if (alpha < 0.01f)
        discard;
    return float4(0.08f, 0.10f, 0.12f, alpha);
}

// ---------------------------------------------------------------------------------------------
// Water table seen through the ground (SCENE_GW_XRAY): translucent surface at H = z_sub + h_gw (capped by the
// ground), drawn without depth test after the ground and before the water
// ---------------------------------------------------------------------------------------------
struct TableVarying
{
    float4 pos   : SV_Position;
    float3 disp  : DISP;
    float  head  : GWHEAD;
    float  depth : TABLEDEPTH;     // below the ground (m)
};

[shader("vertex")]
TableVarying gwTableVert(uint vid : SV_VertexID)
{
    float H = min(gw_head(vid), zb[vid]);
    TableVarying o;
    o.disp  = display_pos(vid / P.ny, vid % P.ny, H);
    o.pos   = project(o.disp);
    o.head  = H;
    o.depth = zb[vid] - H;
    return o;
}

[shader("fragment")]
float4 gwTableFrag(TableVarying v) : SV_Target
{
    if (cropped_out(v.disp))
        discard;
    // lighter where the table is close to the ground, contour lines of the table when they are on
    float3 col = lerp(float3(0.35f, 0.75f, 1.0f), float3(0.05f, 0.25f, 0.75f), saturate(v.depth / max(SC.depth_ref, 1e-3f)));
    float  a   = 0.45f;
    if ((SC.flags & SCENE_GW_CONTOURS) != 0 && SC.gw.x > 0.0f)
    {
        float x    = v.head / SC.gw.x;
        float d    = abs(frac(x + 0.5f) - 0.5f) / max(fwidth(x), 1e-6f);
        float line = 1.0f - sstep(0.6f, 1.6f, d);
        col = lerp(col, float3(0.02f, 0.08f, 0.3f), line);
        a   = max(a, 0.8f * line);
    }
    return float4(col, a);
}
