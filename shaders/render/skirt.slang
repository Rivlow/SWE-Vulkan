// Model base ("socle"): vertical walls along the 4 sides, earth down to z_base and the water slice
// from the free surface down to the bed. Generated from SV_VertexID, 6 vertices per perimeter segment.
// Port of the skirt part of render() in ui/renderer.py.
import params;
import scene;
import palette;

struct SkirtVarying
{
    float4 pos   : SV_Position;
    float3 color : COLOR;
};

// p-th cell around the domain (counter-clockwise) -> (i, j) and outward normal (display frame)
void perimeter(uint p, out uint i, out uint j, out float3 n)
{
    uint nx = P.nx, ny = P.ny;
    if (p < nx)                  { i = p;                      j = 0;                       n = float3(0, 0, 1);  }  // south
    else if (p < nx + ny)        { i = nx - 1;                 j = p - nx;                  n = float3(1, 0, 0);  }  // east
    else if (p < 2 * nx + ny)    { i = nx - 1 - (p - nx - ny); j = ny - 1;                  n = float3(0, 0, -1); }  // north
    else                         { i = 0;                      j = ny - 1 - (p - 2 * nx - ny); n = float3(-1, 0, 0); } // west
}

// corner c of the quad of segment p: top/bottom of perimeter cells p and p+1
void corner(uint vid, out uint p, out bool top)
{
    uint seg  = vid / 6;
    uint c    = vid % 6;                 // (p,top) (p,bot) (q,bot) (p,top) (q,bot) (q,top)
    uint nper = 2 * (P.nx + P.ny);
    bool useQ = c == 2 || c == 4 || c == 5;
    p   = useQ ? (seg + 1) % nper : seg;
    top = c == 0 || c == 3 || c == 5;
}

[shader("vertex")]
SkirtVarying skirtBedVert(uint vid : SV_VertexID)
{
    uint p; bool top;
    corner(vid, p, top);
    uint i, j; float3 n;
    perimeter(p, i, j, n);
    float z     = zb[idx(i, j)];
    float shade = 0.55f + 0.45f * max(dot(n, SC.sun.xyz), 0.0f) + 0.1f;
    float stripe = 0.9f + 0.1f * sin(40.0f * z / SC.depth_ref);      // faint strata

    SkirtVarying o;
    o.pos   = project(display_pos(i, j, top ? z : SC.z_base));
    o.color = top ? SOIL * 1.15f * shade * stripe : SOIL * 0.55f * shade;
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
    o.pos   = project(display_pos(i, j, top ? eta : z));
    o.color = top ? lerp(SHALLOW, DEEP, 0.4f) * shade : DEEP * shade;
    return o;
}

[shader("fragment")]
float4 skirtFrag(SkirtVarying v) : SV_Target
{
    return float4(v.color, 1.0f);
}
