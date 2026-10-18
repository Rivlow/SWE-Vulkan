#pragma once
// Vocabulary of the simulation model, shared by the case description, the solver and the application.
// The numeric values are also the GPU encoding (see shaders/solver/params.slang, checked in SolverParams.h).
#include <cstdint>

namespace swe {

enum class BcKind : uint32_t
{
	Wall   = 0,
	Inflow = 1,        // discharge (m3/s) spread along the side
	Level  = 2,        // imposed free-surface level
};

struct BoundaryCondition
{
	BcKind kind  = BcKind::Wall;
	double value = 0.0;        // inflow discharge (m3/s) or free-surface level
};

// Per-cell conditions applied at the end of each step
enum class ZoneKind : uint32_t
{
	None   = 0,
	Source = 1,        // discharge (m3/s) spread over the zone
	Sink   = 2,        // outlet
	Level  = 3,        // fixed free-surface elevation (relative to z_ref)
};

// Water block edits on a cell rectangle
enum class WaterEdit : uint32_t
{
	Depth = 0,        // h = value
	Level = 1,        // h = max(value - zb, 0)
	Add   = 2,        // h += value
	Clear = 3,
};

// Inclusive rectangle of cells (i along x, j along y); empty when i1 < i0 or j1 < j0
struct CellRect
{
	uint32_t i0 = 1, j0 = 0, i1 = 0, j1 = 0;
	bool     empty() const { return i1 < i0 || j1 < j0; }
};

constexpr uint32_t MAX_ZONES        = 16;
constexpr uint32_t MAX_PROBES       = 8;
constexpr int      MAX_DRAIN_PASSES = 8;

}        // namespace swe
