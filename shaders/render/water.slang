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
};

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
    o.cval     = SC.color_mode != COLOR_REALISTIC ? colormap_value(vid, h) : 0.0f;
    return o;
}

// Wireframe overlay ("Maillage"): same surface as waterVert, position only (fragment: terrain.slang wireFrag)
[shader("vertex")]
float4 waterWireVert(uint vid : SV_VertexID) : SV_Position
{
    return waterVert(vid).pos;
}

struct WaterOut
{
    [[vk::location(0), vk::index(0)]] float4 add : SV_Target0;   // B
    [[vk::location(0), vk::index(1)]] float4 mul : SV_Target1;   // K, multiplies the bed already in the framebuffer
};

[shader("fragment")]
WaterOut waterFrag(WaterVarying v)
{
    float3 sun = SC.sun.xyz;
    float3 n   = normalize(v.n);
    float  h   = max(v.h, 0.0f);
    float  ao = v.ao, sun_vis = v.shadow;

    WaterOut o;
    if (SC.color_mode != COLOR_REALISTIC)
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
    float3 K = (1.0f - fm) * (1.0f - fres) * shade * (1.0f - opacity) * (0.45f + SHALLOW);
    float3 B = (1.0f - fm) * ((1.0f - fres) * shade * opacity * tint + fres * SKY + glint) + fm * foam_col;
    o.add = float4(B, 1.0f);
    o.mul = float4(K, 0.0f);
    return o;
}
