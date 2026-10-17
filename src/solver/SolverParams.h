#pragma once
// Host mirror of shaders/solver/params.slang - keep both in sync.
#include <array>
#include <cstdint>

namespace swe {

enum SolverFlags : uint32_t
{
	FLAG_SINK     = 1,
	FLAG_INFLOW   = 2,
	FLAG_FRICTION = 4,
};

enum class BcKind : uint32_t
{
	Wall   = 0,
	Inflow = 1,
	Level  = 2,
};

struct SolverParams        // std140 uniform buffer
{
	uint32_t nx = 0, ny = 0, n_cells = 0, flags = 0;
	float    dx = 0, dy = 0, g = 0, cfl_dx = 0;
	float    h_eps = 0, h_dry = 0, h_dry4 = 0, eps2 = 0;
	float    g_n2 = 0, _pad0 = 0, _pad1 = 0, _pad2 = 0;
	std::array<uint32_t, 4> bc_kind{};        // west, east, south, north
	std::array<float, 4>    bc_value{};
};
static_assert(sizeof(SolverParams) == 96);

struct StagePC             // push constants
{
	uint32_t             n_coefs = 0, stage = 0, _pad0 = 0, _pad1 = 0;
	std::array<float, 8> coef{};
};
static_assert(sizeof(StagePC) == 48);

// Slots of the `scalars` buffer (uint32 each, floats stored as bits).
enum ScalarSlot : uint32_t
{
	S_SMAX_BITS = 0,
	S_SMAX      = 1,
	S_DT        = 2,
	S_TIME      = 3,
	S_NSTEPS    = 4,
	S_VOLUME    = 16,
	S_WET_CELLS = 17,
	S_HMAX      = 18,
	S_UMAX      = 19,
	S_PROBES    = 32,        // MAX_PROBES x (h, u, v, eta)
	S_COUNT     = 64,
};

constexpr uint32_t MAX_PROBES = 8;

struct ProbeValue
{
	float h = 0, u = 0, v = 0, eta = 0;
};

// Values read back from the GPU once per frame.
struct SolverScalars
{
	float    s_max   = 0;
	float    dt      = 0;
	float    t       = 0;
	uint32_t n_steps = 0;
	// diagnostics
	float volume = 0, wetCells = 0, hMax = 0, uMax = 0;
	std::array<ProbeValue, MAX_PROBES> probes{};
};

}        // namespace swe
