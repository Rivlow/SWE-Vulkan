#pragma once
// Vocabulary of the simulation model, shared by the case description, the solver and the application.
// The numeric values are also the GPU encoding (see shaders/solver/params.slang, checked in SolverParams.h).
#include <cmath>
#include <cstdint>
#include <vector>

namespace swe {

enum class BcKind : uint32_t
{
	Wall   = 0,
	Inflow = 1,        // discharge (m3/s) spread along the side
	Level  = 2,        // imposed free-surface level
	Free   = 3,        // free (transmissive) outflow: zero-gradient ghost state
	Depth  = 4,        // imposed water depth (m)
};

// Cell sides, also the face encoding of the weak conditions: face = cell * 4 + side
enum Side : uint32_t
{
	SIDE_WEST  = 0,
	SIDE_EAST  = 1,
	SIDE_SOUTH = 2,
	SIDE_NORTH = 3,
};
inline uint32_t faceOf(uint32_t cell, Side side) { return cell * 4 + side; }
inline uint32_t faceCell(uint32_t face) { return face >> 2; }
inline Side     faceSide(uint32_t face) { return static_cast<Side>(face & 3); }

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

// Arbitrary set of cells: one byte per cell (index i * ny + j), 1 = in the set. Empty vector = no cell.
using CellMask = std::vector<uint8_t>;

// Depth of still water whose free surface is `level` above a bed `z`, such that h + z == level exactly in float
// (the solver keeps such a lake at rest to the bit; see shaders/solver/reconstruction.slang depth_for_level).
inline float depthForLevel(float level, float z)
{
	float h = level - z;
	if (!(h > 0.0f))
		return 0.0f;
	for (int k = 0; k < 4 && h + z != level; ++k)
		h = std::nextafter(h, h + z < level ? 2.0f * h + 1.0f : 0.0f);
	return h > 0.0f ? h : 0.0f;
}

constexpr uint32_t MAX_ZONES        = 16;
constexpr uint32_t MAX_FACE_BC      = 16;        // groups of weak conditions on cell faces
constexpr uint32_t MAX_PROBES       = 8;
constexpr int      MAX_DRAIN_PASSES = 8;

}        // namespace swe
