#pragma once
// Host mirror of shaders/solver/params.slang - keep both in sync.
#include "cases/CaseTypes.h"

#include <array>
#include <cstdint>

namespace swe {

struct CaseData;

enum SolverFlags : uint32_t
{
	FLAG_ZONES    = 1,
	FLAG_INFLOW   = 2,
	FLAG_FRICTION = 4,
	FLAG_EXACT_INFLOW = 8,
	FLAG_GROUNDWATER  = 16,        // aquifer under the surface
	FLAG_GW_LATERAL   = 32,        //   with lateral flow (some kh > 0)
	FLAG_GW_WOLFGPU   = 64,        //   wolfgpu convention for the vertical exchange
	FLAG_SOURCES      = 128,       // infiltration / rain chronologies
};

// The model enums are also the GPU encoding (params.slang BC_*, ZONE_*, WATER_*)
static_assert(uint32_t(BcKind::Inflow) == 1 && uint32_t(BcKind::Level) == 2);
static_assert(uint32_t(ZoneKind::Source) == 1 && uint32_t(ZoneKind::Sink) == 2 && uint32_t(ZoneKind::Level) == 3);
static_assert(uint32_t(WaterEdit::Level) == 1 && uint32_t(WaterEdit::Add) == 2 && uint32_t(WaterEdit::Clear) == 3);

struct Zone
{
	uint32_t kind   = 0;
	uint32_t ncells = 0;
	float    value  = 0;
	float    _pad   = 0;
};

struct SolverParams        // std140 uniform buffer
{
	uint32_t nx = 0, ny = 0, n_cells = 0, flags = 0;
	float    dx = 0, dy = 0, g = 0, cfl_dx = 0;
	float    h_eps = 0, h_dry = 0, h_dry4 = 0, eps2 = 0;
	float    g_n2 = 0, dt_max = 1e30f, bc_fill = 0.5f, gw_dt_coef = 0;
	std::array<uint32_t, 4> bc_kind{};        // west, east, south, north
	std::array<float, 4>    bc_value{};
	std::array<float, 4>    bc_zmin{};        // lowest bed elevation of each side
};
static_assert(sizeof(SolverParams) == 112);

// GPU parameters of a case (pure function: constants folded in double, then rounded to f32)
SolverParams buildSolverParams(CaseData const &c);

// euler_stage flags
enum StageFlags : uint32_t
{
	STAGE_FIRST = 1,        // saves Q^n into Q_old (multi-stage schemes)
	STAGE_LAST  = 2,        // friction, zones, s_max of the new state
};

struct StagePC             // push constants (see params.slang)
{
	uint32_t             flags = 0;        // euler_stage: StageFlags
	uint32_t             pass  = 0;        // drain_alpha: Jacobi pass; euler_stage: passes done (0 = limiter off)
	uint32_t             mode  = 0;        // set_water: WaterEdit
	uint32_t             _pad0 = 0;
	std::array<float, 8> coef{};           // euler_stage: (a, b); set_water: i0 j0 i1 j1 value
};
static_assert(sizeof(StagePC) == 48);

// Descriptor bindings of the solver set (params.slang [[vk::binding(b, 0)]])
enum SolverBinding : uint32_t
{
	B_PARAMS = 0,        // uniform buffer, every other binding is a storage buffer
	B_SCALARS,
	B_ZB,
	B_Q,
	B_Q_OLD,
	B_ALPHA_A,
	B_F_L,
	B_F_R,
	B_G_D,
	B_G_U,
	B_ALPHA_B,
	B_COND,
	B_INFLOW,
	B_PROBE_CELLS,
	B_DIAG,
	B_ZONES,
	B_MASS_FLUX,
	B_TILE_WET,
	B_TILE_LIST,
	B_TILE_ARGS,
	B_TILE_SEED,
	B_SOIL,
	B_GW,
	B_GW_AUX,
	B_DIAG_GW,
	B_SRC_ZONES,
	B_CHRONO,
	B_SRC_RATES,
	B_SRC_CUM,
	B_PROFILE_CELLS,
	B_PROFILE_OUT,
	B_COUNT,
};

// Slots of the `scalars` buffer (uint32 each, floats stored as bits).
enum ScalarSlot : uint32_t
{
	S_SMAX_BITS = 0,
	S_SMAX      = 1,
	S_DT        = 2,
	S_TIME      = 3,        // f32 sum of the steps (effects)
	S_NSTEPS    = 4,
	S_TIME_LO   = 5,        // exact clock, 64-bit fixed point: t = S_TIME_HI 2^-8 + S_TIME_LO 2^-40 (s)
	S_TIME_HI   = 6,
	S_GW_D_BITS = 7,        // aquifer diffusivity accumulator (atomic max), consumed by finalize_dt
	S_GW_D      = 8,
	S_GW_VOLUME = 9,        // groundwater diagnostics: aquifer volume, cumulated infiltration / exfiltration (m3),
	S_GW_INFIL  = 10,       //   net exchange of the last step (m3/s, > 0 = infiltration)
	S_GW_EXFIL  = 11,
	S_GW_RATE   = 12,
	S_SRC_NET   = 13,       // chronologies: net volume added to the surface since t = 0 (m3)
	S_SRC_RAIN  = 14,       //   rain volume since t = 0 (m3)
	S_VOLUME    = 16,
	S_WET_CELLS = 17,
	S_HMAX      = 18,
	S_UMAX      = 19,
	S_LIMITED   = 20,        // cells limited (alpha < 1) on the last stage of the last step
	S_CLAMP_N   = 21,        // cells clamped to h = 0 since t = 0
	S_CLAMP_MAX = 22,        // largest clamped |h| since t = 0 (float bits)
	S_PROBES    = 32,        // MAX_PROBES x (h, u, v, eta)
	S_GW_PROBES = 64,        // MAX_PROBES x (water table elevation, exchange rate m/s), groundwater cases
	S_COUNT     = 80,
};
static_assert(S_PROBES + 4 * MAX_PROBES <= S_GW_PROBES && S_GW_PROBES + 2 * MAX_PROBES <= S_COUNT);

constexpr uint32_t MAX_PROFILE = 2048;        // params.slang

// One point of a profile (read back each frame): ground, water depth, water table, substratum
struct ProfilePoint
{
	float zb = 0, h = 0, head = 0, zSub = 0;
};

struct ProbeValue
{
	float h = 0, u = 0, v = 0, eta = 0;
	float gwHead = 0, gwRate = 0;        // groundwater: water table elevation, exchange rate (m/s, > 0 = infiltration)
};

// Values read back from the GPU once per frame.
struct SolverScalars
{
	float    s_max   = 0;
	float    dt      = 0;
	double   t       = 0;        // exact sum of the time steps (to 2^-40 s per step)
	uint32_t n_steps = 0;
	// diagnostics
	float volume = 0, wetCells = 0, hMax = 0, uMax = 0;
	// drain limiter: cells limited on the last step; cells clamped to 0 and largest clamp since t = 0
	uint32_t limitedCells = 0, clampCells = 0;
	float    clampMax = 0;
	// groundwater: aquifer volume, cumulated infiltration and exfiltration (m3), exchange rate (m3/s), diffusivity
	float gwVolume = 0, gwInfil = 0, gwExfil = 0, gwRate = 0, gwDiffusivity = 0;
	// infiltration / rain chronologies: net volume added to the surface, rain volume since t = 0 (m3)
	float srcNet = 0, srcRain = 0;
	std::array<ProbeValue, MAX_PROBES> probes{};
};

}        // namespace swe
