// Shared solver state: parameters, buffers and indexing.
// Every resource has an explicit binding so all entry points of solver.spv share one descriptor set layout.
// Mirror of src/solver/SolverParams.h - keep both in sync.
module params;

public static const uint FLAG_SINK     = 1;
public static const uint FLAG_INFLOW   = 2;
public static const uint FLAG_FRICTION = 4;

public static const uint BC_WALL   = 0;
public static const uint BC_INFLOW = 1;
public static const uint BC_LEVEL  = 2;

public struct SolverParams          // std140, 96 bytes
{
    public uint  nx;
    public uint  ny;
    public uint  n_cells;
    public uint  flags;
    public float dx;
    public float dy;
    public float g;
    public float cfl_dx;            // cfl * min(dx, dy), folded in double on the host like Taichi does
    public float h_eps;
    public float h_dry;
    public float h_dry4;            // h_dry^4
    public float eps2;              // Venkatakrishnan epsilon^2
    public float g_n2;              // g * manning_n^2
    public float _pad0;
    public float _pad1;
    public float _pad2;
    public uint4  bc_kind;          // west, east, south, north: BC_WALL / BC_INFLOW / BC_LEVEL
    public float4 bc_value;         // inflow discharge (m3/s) or free-surface level
};

public struct StagePC               // push constants, 48 bytes
{
    public uint   n_coefs;          // combine: number of Runge-Kutta coefficients
    public uint   stage;            // residual: which K slot to write
    public uint   _pad0;
    public uint   _pad1;
    public float4 coef[2];          // up to 8 coefficients (dopri5 needs 7)
};

// scalars[] slots (uint, floats stored with asuint)
public static const uint S_SMAX_BITS = 0;   // atomic max accumulator, reset by finalize_dt
public static const uint S_SMAX      = 1;
public static const uint S_DT        = 2;
public static const uint S_TIME      = 3;
public static const uint S_NSTEPS    = 4;

[[vk::binding(0, 0)]] public ConstantBuffer<SolverParams> P;
[[vk::binding(1, 0)]] public RWStructuredBuffer<uint>     scalars;
[[vk::binding(2, 0)]] public StructuredBuffer<float>      zb;
[[vk::binding(3, 0)]] public RWStructuredBuffer<float4>   Q;       // (h, hu, hv, 0)
[[vk::binding(4, 0)]] public RWStructuredBuffer<float4>   Q_old;
[[vk::binding(5, 0)]] public RWStructuredBuffer<float4>   K;       // K[stage * n_cells + cell]
[[vk::binding(6, 0)]] public RWStructuredBuffer<float4>   F_L;     // x-faces (nx+1, ny): flux seen by the left cell
[[vk::binding(7, 0)]] public RWStructuredBuffer<float4>   F_R;     //                     flux seen by the right cell
[[vk::binding(8, 0)]] public RWStructuredBuffer<float4>   G_D;     // y-faces (nx, ny+1): flux seen by the lower cell
[[vk::binding(9, 0)]] public RWStructuredBuffer<float4>   G_U;     //                     flux seen by the upper cell
[[vk::binding(10, 0)]] public RWStructuredBuffer<float2>  S_b;     // centred bed-slope source (Sx, Sy)
[[vk::binding(11, 0)]] public StructuredBuffer<uint>      sink;    // 1 = outlet cell (1-element dummy when absent)
// Own buffer (not a scalars slot): bc_inflow_norm runs unsynchronised next to finalize_dt.
[[vk::binding(12, 0)]] public RWStructuredBuffer<float>   inflow;  // sum of h^(5/3) * width per side

[[vk::push_constant]] public ConstantBuffer<StagePC> pc;

// Row-major (nx, ny) like numpy C order: j varies fastest.
public uint idx(uint i, uint j)  { return i * P.ny + j; }
public uint fidx(uint i, uint j) { return i * P.ny + j; }          // x-face (nx+1, ny)
public uint gidx(uint i, uint j) { return i * (P.ny + 1) + j; }    // y-face (nx, ny+1)

public float stage_coef(uint m) { return pc.coef[m >> 2][m & 3]; }
