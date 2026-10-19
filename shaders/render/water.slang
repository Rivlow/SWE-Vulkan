// Free-surface mesh, port of the water part of render() in ui/renderer.py.
//
// The Python viewer faked transparency per vertex: col = mix(bed * (0.45 + SHALLOW), tint, opacity),
// then shading, sky reflection, sun glint and foam. That chain is linear in the bed colour, so it is
// computed here exactly with dual-source blending: framebuffer = bed * K + B, where "bed" is the terrain
// pixel actually rendered behind (per pixel, not per vertex). Blend: src0 * ONE + dst * SRC1_COLOR.
import params;
import reconstruction;
import effects_data;
import scene;
import noise;
import palette;
import colormap;

struct WaterVarying
{
    float4 pos      : SV_Position;
    float3 disp     : DISP;
    float3 n        : NORMAL;
    float  h        : DEPTH;
    float  wet_blur : WETBLUR;
    float  foam     : FOAMF;       // advected foam field
    float  turb     : TURB;        // advected turbidity
    float  shadow   : SHADOW;
    float  ao       : AO;
    float  cval     : CMAPVALUE;   // quantity shown by the colour map
    nointerpolation uint2 block : BLOCK;   // raw cells: (owner cell, side + 1) of a wall, (0, 0) otherwise
};

// colour map of a surface quantity (groundwater quantities colour the ground, the water stays realistic)
bool water_color_mode()
{
    return SC.color_mode != COLOR_REALISTIC && SC.color_mode < COLOR_GW_TABLE_DEPTH;
}

float colormap_value(uint k, float h)
{
    float4 q = Q[k];
    float  u = velocity(h, q.y), v = velocity(h, q.z);
    float  speed = sqrt(u * u + v * v);
    switch (SC.color_mode)
    {
    case COLOR_DEPTH:     return h;
    case COLOR_SPEED:     return speed;
    case COLOR_FROUDE:    return speed / sqrt(P.g * max(h, 1e-3f));
    case COLOR_ELEVATION: return zb[k] + h;
    default:              return 0.0f;
    }
}

[shader("vertex")]
WaterVarying waterVert(uint vid : SV_VertexID)
{
    int   i = int(vid / P.ny), j = int(vid % P.ny);
    float z = zb[vid];
    float h = Q[vid].x;

    // eta = z + h, hidden under the bed when dry; next to a dry wall the hidden vertex also goes below the
    // neighbouring water (otherwise the edge triangles climb the wall like a veil)
    float eta = z + h;
    float sink = 0.05f * SC.depth_ref;
    if (h < 1e-3f)
    {
        eta = z - sink;
        int2 nb[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
        for (int k = 0; k < 4; ++k)
        {
            int2 c = clamp_ij(i + nb[k].x, j + nb[k].y);
            uint m = idx(c.x, c.y);
            float hn = Q[m].x;
            if (hn >= 1e-3f)
                eta = min(eta, zb[m] + hn - sink);
        }
    }
    float dex = (water_height(i + 1, j) - water_height(i - 1, j)) / (2.0f * P.dx) * SC.z_scale;
    float dey = (water_height(i, j + 1) - water_height(i, j - 1)) / (2.0f * P.dy) * SC.z_scale;

    WaterVarying o;
    o.disp = display_pos(i, j, eta);
    o.pos  = project(o.disp);
    o.n    = float3(-dex, 1.0f, dey);
    o.h        = h;
    o.wet_blur = wet_blur[vid];
    o.foam     = foam[vid];
    o.turb     = turb[vid];
    o.shadow   = shadow[vid];
    o.ao       = ao[vid];
    o.cval     = water_color_mode() ? colormap_value(vid, h) : 0.0f;
    o.block    = uint2(0, 0);
    return o;
}

// Top of the water column of cell k: its free surface when wet, else its bed
float column_top(uint k)
{
    float h = Q[k].x;
    return zb[k] + (h >= 1e-3f ? h : 0.0f);
}

// Raw cells: flat water columns from the bed to the free surface (18 vertices per cell, scene.slang
// block_vertex). The top of a column is its free surface when wet, else its bed (so a wall between a wet cell
// and a dry one goes down to the dry bed); dry tops are collapsed.
[shader("vertex")]
WaterVarying waterBlockVert(uint vid : SV_VertexID)
{
    uint  cell = vid / 18;
    uint  i = cell / P.ny, j = cell % P.ny;
    bool  has_east = i + 1 < P.nx, has_north = j + 1 < P.ny;
    float z0 = column_top(cell);
    float ze = has_east ? column_top(cell + P.ny) : z0, zn = has_north ? column_top(cell + 1) : z0;
    bool  wet0 = Q[cell].x >= 1e-3f;
    bool  wete = has_east && Q[cell + P.ny].x >= 1e-3f, wetn = has_north && Q[cell + 1].x >= 1e-3f;
    // walls only next to water, and only for real steps: between two wet cells the free surface differs by a
    // few millimetres almost everywhere, and those slivers would tile the water with shaded lines
    float eps = 2e-3f;
    bool  stepe = (wet0 || wete) && !(wet0 && wete && abs(ze - z0) < eps);
    bool  stepn = (wet0 || wetn) && !(wet0 && wetn && abs(zn - z0) < eps);
    BlockVertex b = block_vertex(vid, cell, z0, stepe ? ze : z0, has_east, stepn ? zn : z0, has_north, !wet0);
    uint  k = b.owner;
    float h = Q[k].x;

    WaterVarying o;
    o.disp     = b.disp;
    o.pos      = project(o.disp);
    o.n        = b.n;
    o.h        = h;
    o.wet_blur = wet_blur[k];
    o.foam     = foam[k];
    o.turb     = turb[k];
    o.shadow   = shadow[k];
    o.ao       = ao[k];
    o.cval     = water_color_mode() ? colormap_value(k, h) : 0.0f;
    o.block    = uint2(k, b.wall ? b.side + 1 : 0);
    return o;
}

struct WaterOut
{
    [[vk::location(0), vk::index(0)]] float4 add : SV_Target0;   // B
    [[vk::location(0), vk::index(1)]] float4 mul : SV_Target1;   // K, multiplies the bed already in the framebuffer
};

[shader("fragment")]
WaterOut waterFrag(WaterVarying v)
{
    if (cropped_out(v.disp))
        discard;
    float3 sun = SC.sun.xyz;
    float3 n   = normalize(v.n);
    // block walls are seen from either side: face the camera
    if (v.block.y != 0 && dot(n, SC.cam_pos.xyz - v.disp) < 0.0f)
        n = -n;
    float  h   = max(v.h, 0.0f);
    float  ao = v.ao, sun_vis = v.shadow;

    WaterOut o;
    if (water_color_mode())
    {
        // scientific view: colour map of the chosen quantity, lightly shaded so the relief of the surface reads
        float  t     = (v.cval - SC.cmap_min) / max(SC.cmap_max - SC.cmap_min, 1e-12f);
        float  shade = (0.78f + 0.3f * max(dot(n, sun), 0.0f) * sun_vis) * (0.75f + 0.25f * ao);
        float  a     = SC.cmap_opacity * sstep(0.0f, 0.02f, h);
        o.add = float4(a * turbo(t) * shade, 1.0f);
        o.mul = float4((1.0f - a).xxx, 0.0f);
        return o;
    }

    // absorption (Beer-Lambert): the tinted bed shows through all the more as the water is shallow;
    // turbid water is brown and much less transparent (optional)
    float  T       = (E.flags & EFFECTS_TURBIDITY) != 0 ? saturate(v.turb) : 0.0f;
    float3 tint    = lerp(SHALLOW, DEEP, 1.0f - exp(-h / (0.6f * SC.depth_ref)));
    tint           = lerp(tint, MUD, 0.85f * T);
    float  opacity = 1.0f - exp(-h / (SC.water_clarity * (1.0f - 0.85f * T)));
    // thin films (a few cm) would be invisible: give them a floor so flooding fronts can be followed
    if ((SC.flags & SCENE_THIN_WATER) != 0)
        opacity = max(opacity, 0.45f * sstep(0.002f, 0.03f, h));
    // groundwater map on the ground: light, nearly clear water so the map reads through it
    float bed_tint = 0.45f;
    if (gw_color_mode())
    {
        opacity  *= 0.25f;
        bed_tint = 0.85f;
    }
    float  shade   = (0.8f + 0.35f * max(dot(n, sun), 0.0f) * sun_vis) * (0.6f + 0.4f * ao);

    // sky reflection (Schlick Fresnel) + sun glint (Blinn-Phong), none in the shade
    float3 view = normalize(SC.cam_pos.xyz - v.disp);
    float  wet_ = sstep(0.0f, 0.05f, h);                         // thin films: no reflection, the bed shows
    float  fres = min(0.02f + 0.98f * pow(1.0f - max(dot(n, view), 0.0f), 5.0f), 0.6f) * wet_;
    float3 hv   = normalize(view + sun);
    float3 glint = float3(1.0f, 0.97f, 0.9f) * 0.7f * wet_ * sun_vis * pow(max(dot(n, hv), 0.0f), 80.0f);

    // foam: field carried by the flow, textured at sub-cell scale, plus a fringe on the water's edge
    float grain = 0.55f + 0.45f * value_noise(9.0f * v.disp.x, 9.0f * v.disp.z);
    float foam  = v.foam * grain;
    foam = max(foam, sstep(0.85f, 0.55f, v.wet_blur) * sstep(0.06f, 0.0f, h) * 0.35f);
    float fm    = min(foam, 0.8f);
    float3 foam_col = FOAM * (0.75f + 0.25f * sun_vis);

    // out = mix(mix(shade * mix(bed * (0.45 + SHALLOW), tint, opacity), SKY, fres) + glint, foam_col, fm)
    //     = bed * K + B
    float3 K = (1.0f - fm) * (1.0f - fres) * shade * (1.0f - opacity) * min(bed_tint + SHALLOW, 1.0f);
    float3 B = (1.0f - fm) * ((1.0f - fres) * shade * opacity * tint + fres * SKY + glint) + fm * foam_col;
    // selection / borders / zone overlays on the water too (applied to the additive part, where they change it)
    float3 base = B + K * 0.5f;
    float3 over = v.block.y != 0 ? wall_overlays(base, v.block.x, v.block.y - 1) : overlays(base, v.disp);
    if (any(over != base))
    {
        B = over;
        K = K * 0.3f;
    }
    o.add = float4(B, 1.0f);
    o.mul = float4(K, 0.0f);
    return o;
}
