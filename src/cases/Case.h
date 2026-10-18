#pragma once
#include "cases/CaseTypes.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace swe {

// Everything the solver and the renderer need to start a simulation.
// Arrays are (nx, ny) in numpy C order: index = i * ny + j, i along x (east), j along y (north).
struct CaseData
{
	std::string name;
	std::string label;          // human-readable title (GUI)
	double      z_ref = 0.0;    // elevations are relative to z_ref (f32 precision), in m
	int         epsg  = 31370;  // metric projection of geographic cases (Lambert 72, UTM zone...)
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
	// drain limiter: Jacobi passes per stage (0 = off: h < 0 is clamped, which creates mass)
	int         drain_passes = 4;
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

	// Unconfined aquifer under the surface (wolfgpu's groundwater layer): vertical Darcy exchange with the
	// surface water, lateral Boussinesq flow. Per-cell arrays, all of size cells() when enabled.
	struct Groundwater
	{
		bool               enabled = false;
		// wolfgpu convention, for comparisons: vertical exchange compares zb + h with h_gw itself, z_sub = zb,
		// Sy = 1, no storage limit. Otherwise the water table H = z_sub + h_gw lives in the soil (H <= zb).
		bool               wolfgpu = false;
		std::vector<float> kv;        // vertical conductivity (m/s): exchange rate with the surface
		std::vector<float> kh;        // horizontal conductivity (m/s): transmissivity T = kh h_gw
		std::vector<float> sy;        // specific yield (drainable porosity), (0, 1]
		std::vector<float> z_sub;     // impervious substratum elevation (relative to z_ref)
		std::vector<float> h0;        // initial saturated thickness above z_sub (m)
	};
	Groundwater gw;

	// wolfgpu's infiltration and rain chronologies: per-cell zone maps (1-based, 0 = none) and a table of
	// discharges per zone over time. A value is the discharge of the whole zone (m3/s), spread uniformly over its
	// cells; positive adds water to the surface (infiltration < 0 removes it, at most what is there).
	struct Chronology
	{
		std::vector<uint16_t>            zones;        // per cell, empty = no chronology
		std::vector<double>              t;            // start time of each line (s), increasing
		std::vector<std::vector<double>> q;            // q[line][zone - 1] (m3/s)
		bool                             lerp = false; // linear interpolation between lines (else steps)
		bool active() const { return !zones.empty() && !t.empty(); }
		size_t zoneCount() const { return q.empty() ? 0 : q[0].size(); }
	};
	Chronology infiltration, rain;

	size_t cells() const { return size_t(nx) * ny; }
	size_t index(uint32_t i, uint32_t j) const { return size_t(i) * ny + j; }
};

}        // namespace swe
