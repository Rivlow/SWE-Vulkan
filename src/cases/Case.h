#pragma once
#include "solver/SolverParams.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace swe {

struct BoundaryCondition
{
	BcKind kind  = BcKind::Wall;
	double value = 0.0;        // inflow discharge (m3/s) or free-surface level
};

// Everything the solver and the renderer need to start a simulation.
// Arrays are (nx, ny) in numpy C order: index = i * ny + j, i along x (east), j along y (north).
struct CaseData
{
	std::string name;
	std::string label;          // human-readable title (GUI)
	double      z_ref = 0.0;    // elevations are relative to z_ref (f32 precision), in m
	uint32_t    nx = 0, ny = 0;
	double      Lx = 0, Ly = 0;
	double      dx() const { return Lx / nx; }
	double      dy() const { return Ly / ny; }

	// physics / numerics (main.py defaults)
	double      g         = 9.81;
	double      cfl       = 0.4;
	double      h_eps     = 1e-4;
	double      h_dry     = 1e-3;
	double      manning_n = 0.0;
	std::string scheme    = "ssprk2";
	std::array<BoundaryCondition, 4> bc{};        // west, east, south, north
	// inflow sides: false = mass flux exactly the imposed discharge (default); true = HLLC flux of the ghost
	// state as in the Python reference (used to validate against it)
	bool legacyInflow = false;

	// rendering hints
	double eta_0     = 1.0;
	double depth_ref = 1.0;
	double z_scale_0 = 2.0;

	// fields
	std::vector<float>    zb;           // bed elevation
	std::vector<float>    h0, hu0, hv0; // initial state
	std::vector<uint32_t> sink;         // outlet cells (empty = none) - legacy, converted to a Sink zone on load
	std::vector<uint8_t>  outside;      // cells outside the useful domain, rendering only (empty = none)

	// per-cell conditions: zones[k] applies to cells with zoneMask == k + 1 (empty = none)
	struct ZoneDef
	{
		ZoneKind    kind = ZoneKind::Source;
		double      value = 0.0;        // m3/s (source) or elevation relative to z_ref (level)
		std::string label;
		uint32_t    i0 = 0, j0 = 0, i1 = 0, j1 = 0;        // rectangle used to create it (GUI)
	};
	std::vector<ZoneDef>  zones;
	std::vector<uint8_t>  zoneMask;

	size_t cells() const { return size_t(nx) * ny; }
};

}        // namespace swe
