// Visual effect fields of ui/renderer.py, one value per cell (index idx(i, j)), plus streamline state.
// Set 1 of both the effects compute pipelines and the render pipelines. Mirror of src/render/Effects.h.
module effects_data;

public static const uint N_LINES = 110;
public static const uint N_PTS   = 24;      // points per streamline

public struct EffectsParams             // std140, 64 bytes
{
    public uint  shore_radius;          // ~12 m of shore, between 1 and 6 cells
    public uint  flags;                 // EFFECTS_TURBIDITY
    public uint  _pad0;
    public uint  _pad1;
    public float L_ref;                 // min(dx, dy): dissipation scales in cells
    public float foam_length;
    public float turb_length;
    public float settle_time;
    public float dry_time;
    public float ds;                    // streamline integration step (m)
    public float z_top;                 // max bed elevation (shadow rays stop above it)
    public float u_min_lines;           // no streamline below this speed (m/s)
    public float Lx;
    public float Ly;
    public float _pad2;
    public float _pad3;
};

public static const uint EFFECTS_TURBIDITY = 1;
public static const uint EFFECTS_GW_LINES  = 2;   // streamlines follow the aquifer flow (drawn in blue on the ground)

public struct Seed
{
    public float2 pos;                  // physical position (m)
    public float  age;                  // real time (s)
    public float  life;
};

public struct LinePoint
{
    public float2 pos;                  // physical position (m)
    public float  eta;                  // free-surface elevation
    public float  intensity;            // 0..1, < 0: dead segment end
};

public struct EffectsState
{
    public float t_prev;                // simulated time at the previous frame
    public float t_frame;               // simulated time elapsed during this frame (0 when paused)
    public float _pad0;
    public float _pad1;
};

[[vk::binding(0, 1)]]  public ConstantBuffer<EffectsParams>   E;
[[vk::binding(1, 1)]]  public RWStructuredBuffer<float>       wet;
[[vk::binding(2, 1)]]  public RWStructuredBuffer<float>       wet_tmp;
[[vk::binding(3, 1)]]  public RWStructuredBuffer<float>       wet_blur;
[[vk::binding(4, 1)]]  public RWStructuredBuffer<float>       shadow;     // 0.35..1: cast sun shadow
[[vk::binding(5, 1)]]  public RWStructuredBuffer<float>       ao;         // ambient occlusion
[[vk::binding(6, 1)]]  public RWStructuredBuffer<float>       turb;
[[vk::binding(7, 1)]]  public RWStructuredBuffer<float>       turb_tmp;
[[vk::binding(8, 1)]]  public RWStructuredBuffer<float>       foam;
[[vk::binding(9, 1)]]  public RWStructuredBuffer<float>       foam_tmp;
[[vk::binding(10, 1)]] public RWStructuredBuffer<float>       wet_mem;    // wet ground drying slowly
[[vk::binding(11, 1)]] public RWStructuredBuffer<float>       flooded;    // maximum flood extent
[[vk::binding(12, 1)]] public RWStructuredBuffer<Seed>        seeds;
[[vk::binding(13, 1)]] public RWStructuredBuffer<LinePoint>   line_pts;   // N_LINES * N_PTS
[[vk::binding(14, 1)]] public RWStructuredBuffer<EffectsState> fx_state;
