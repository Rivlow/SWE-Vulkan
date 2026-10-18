// Streamlines (points computed by effects.slang update_streamlines) drawn as screen-space quads with
// anti-aliased edges, alpha-blended towards the foam colour: same as the Python mix(water, FOAM, intensity),
// applied to the water pixel actually behind.
import params;
import effects_data;
import scene;
import palette;

static const float HALF_WIDTH = 1.25f;     // pixels (the Python lines were 2 px wide)
static const float FEATHER    = 1.0f;      // anti-aliasing ramp, pixels

struct LineVarying
{
    float4 pos       : SV_Position;
    float  across    : ACROSS;     // signed distance from the centre line, pixels
    float  intensity : INTENSITY;
    float3 disp      : DISP;       // display position (cut box)
};

float3 line_display(LinePoint p)
{
    return float3(p.pos.x * SC.S, p.eta * SC.z_scale * SC.S + 0.015f, -p.pos.y * SC.S);
}

// 6 vertices per segment (two triangles); segment m joins points m and m+1 of a line
[shader("vertex")]
LineVarying lineVert(uint vid : SV_VertexID)
{
    uint seg  = vid / 6, corner = vid % 6;
    uint line = seg / (N_PTS - 1), m = seg % (N_PTS - 1);
    LinePoint a = line_pts[line * N_PTS + m];
    LinePoint b = line_pts[line * N_PTS + m + 1];

    LineVarying o;
    if (a.intensity < 0.0f || b.intensity < 0.0f)       // dead segment: degenerate, clipped away
    {
        o.pos = float4(2.0f, 2.0f, 2.0f, 1.0f);
        o.across = 0.0f;
        o.intensity = 0.0f;
        o.disp = float3(0.0f);
        return o;
    }
    static const uint  END[6]  = { 0, 0, 1, 0, 1, 1 };
    static const float SIDE[6] = { -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
    bool   atB = END[corner] == 1;
    float4 ca  = project(line_display(a));
    float4 cb  = project(line_display(b));
    float2 vp  = float2(SC.viewport_w, SC.viewport_h);
    float2 sa  = ca.xy / ca.w * vp, sb = cb.xy / cb.w * vp;
    float2 dir = sb - sa;
    dir = length(dir) > 1e-4f ? normalize(dir) : float2(1.0f, 0.0f);
    float2 perp = float2(-dir.y, dir.x);
    float  w    = HALF_WIDTH + FEATHER;

    float4 c = atB ? cb : ca;
    c.xy += perp * (SIDE[corner] * w) * 2.0f / vp * c.w;
    o.pos       = c;
    o.across    = SIDE[corner] * w;
    o.intensity = atB ? b.intensity : a.intensity;
    o.disp      = line_display(atB ? b : a);
    return o;
}

[shader("fragment")]
float4 lineFrag(LineVarying v) : SV_Target
{
    if (cropped_out(v.disp))
        discard;
    float edge  = 1.0f - saturate((abs(v.across) - HALF_WIDTH) / FEATHER);
    float alpha = saturate(v.intensity) * edge;
    // aquifer streamlines: deep blue on the ground
    return float4((E.flags & EFFECTS_GW_LINES) != 0 ? float3(0.05f, 0.25f, 0.85f) : FOAM, alpha);
}
