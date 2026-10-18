#pragma once
#include "cases/Case.h"
#include "solver/Solver.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace swe {

namespace vkc { class Context; }

// The simulated case and its live GPU state. Single place where a case is edited (parameters, water, zones,
// bed): the CaseData it holds is what Reset restores and Save writes, and every change is forwarded to the
// solver. Used by the interactive application and the batch runs alike; knows nothing of windows or GUI.
class Session
{
  public:
	explicit Session(vkc::Context const &ctx);

	// Loads a case (allocates the GPU state) at t = 0.
	void load(CaseData data);
	bool loaded() const { return case_.nx != 0; }
	// Back to the initial state of the case, t = 0.
	void reset();

	CaseData const &caseData() const { return case_; }
	Solver         &solver() { return solver_; }
	Solver const   &solver() const { return solver_; }
	// Lowest and highest bed elevation (relative to z_ref), cached
	std::pair<float, float> bedRange() const { return bedRange_; }
	// Groundwater cases: soil thickness max (zb - z_sub), lowest substratum, largest kv (m/s), cached
	struct GwStats
	{
		float thicknessMax = 1.0f, zSubMin = 0.0f, kvMax = 0.0f;
	};
	GwStats gwStats() const { return gwStats_; }

	// Runtime parameters (applied to the next submitted steps)
	void setScheme(std::string const &name);        // throws on an unknown scheme
	void setDrainPasses(int passes);
	void setCfl(double cfl);
	void setManning(double n);
	void setBoundary(int side, BoundaryCondition const &bc);

	// Water block at rest in the live state and, if `alsoInitial`, in the initial state (Reset keeps it)
	void editWater(CellRect const &rect, WaterEdit mode, float value, bool alsoInitial);

	// Per-cell zones. addZone returns false when the rectangle is empty or MAX_ZONES are already defined.
	bool addZone(CellRect const &rect, ZoneKind kind, double value, std::string const &label);
	void removeZone(size_t index);
	void setZoneValue(size_t index, double value);

	// Infiltration / rain chronologies (wolfgpu): a new zone takes the cells of a rectangle and the discharge q
	// (m3/s for the whole zone) on every line of the table (one line at t = 0 if there was none).
	enum class ChronologyKind
	{
		Infiltration,
		Rain,
	};
	bool addChronologyZone(ChronologyKind kind, CellRect const &rect, double q);
	void removeChronologyZone(ChronologyKind kind, size_t zone);        // 0-based
	// Replaces the table: increasing times, one discharge per zone and line (m3/s)
	void setChronologyTable(ChronologyKind kind, std::vector<double> t, std::vector<std::vector<double>> q, bool lerp);
	std::vector<double> chronologyZoneAreas(ChronologyKind kind) const;        // m2 per zone

	// Soil / aquifer (groundwater layer). Uniform parameters: vertical and horizontal conductivities (m/s),
	// specific yield, soil thickness above the substratum (m), initial depth of the water table below the ground
	// (m). The case is reloaded at t = 0 (the solver reallocates its buffers).
	struct SoilParams
	{
		double kv = 1e-5, kh = 1e-4, sy = 0.25, depth = 5.0, tableDepth = 2.0;
	};
	// Turns the aquifer on with uniform parameters, or off (nullopt)
	void setGroundwater(std::optional<SoilParams> const &params);
	// Writes the parameters on a rectangle of cells (the aquifer must be on)
	void setSoil(CellRect const &rect, SoilParams const &params);
	// Uniform parameters when the aquifer is on and they are the same on every cell (the GUI's defaults)
	SoilParams soilDefaults() const { return soilDefaults_; }

	// New bed elevations (same grid): still water keeps its surface, the case is reloaded at t = 0.
	void replaceBed(std::vector<float> zb);

  private:
	void applyParameters();
	void applyChronologies();
	CaseData::Chronology &chronology(ChronologyKind kind);
	void updateBedRange();

	Solver                  solver_;
	CaseData                case_;
	std::pair<float, float> bedRange_{0.0f, 0.0f};
	GwStats                 gwStats_;
	SoilParams              soilDefaults_;
};

}        // namespace swe
