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
	// Same on an arbitrary set of cells: the initial state is edited and the live state restored from it (t = 0)
	void editWater(CellMask const &cells, WaterEdit mode, float value);

	// Per-cell zones. addZone returns false when the rectangle / set is empty or MAX_ZONES are already defined.
	bool addZone(CellRect const &rect, ZoneKind kind, double value, std::string const &label);
	bool addZone(CellMask const &cells, ZoneKind kind, double value, std::string const &label);
	void removeZone(size_t index);
	void setZoneValue(size_t index, double value);

	// Weak conditions on cell faces: the `side` faces of the cells of the set that are not shared with another
	// cell of the set (the envelope of the set on that side), unless `allFaces`. Returns false when no face is
	// left or MAX_FACE_BC groups exist. Faces already in a group are moved to the new one.
	bool addFaceBc(CellMask const &cells, Side side, BcKind kind, double value, std::string const &label, bool allFaces = false);
	void removeFaceBc(size_t index);
	void setFaceBc(size_t index, BcKind kind, double value);
	// Cells owning the faces of a group (for the GUI: reselect)
	CellMask faceBcCells(size_t index) const;

	// Bridge decks over a set of cells: soffit and deck top relative to z_ref. A cell belongs to one deck only (the
	// new one takes its cells). Any change restarts the simulation at t = 0.
	bool addBridge(CellMask const &cells, double zRoof, double zDeck, std::string const &label);
	void removeBridge(size_t index);
	void setBridge(size_t index, double zRoof, double zDeck);
	void setBridgeSlot(double slot);

	// Infiltration / rain chronologies (wolfgpu): a new zone takes the cells of a rectangle (or set) and the
	// discharge q (m3/s for the whole zone) on every line of the table (one line at t = 0 if there was none).
	enum class ChronologyKind
	{
		Infiltration,
		Rain,
	};
	bool addChronologyZone(ChronologyKind kind, CellRect const &rect, double q);
	bool addChronologyZone(ChronologyKind kind, CellMask const &cells, double q);
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
	// Writes the parameters on a rectangle / set of cells (the aquifer must be on)
	void setSoil(CellRect const &rect, SoilParams const &params);
	void setSoil(CellMask const &cells, SoilParams const &params);

	// Cells of a rectangle as a mask of the loaded grid (clamped), and the bounding rectangle of a mask
	CellMask maskOf(CellRect const &rect) const;
	CellRect boundsOf(CellMask const &cells) const;
	size_t   countOf(CellMask const &cells) const;
	// Uniform parameters when the aquifer is on and they are the same on every cell (the GUI's defaults)
	SoilParams soilDefaults() const { return soilDefaults_; }

	// New bed elevations (same grid): still water keeps its surface, the case is reloaded at t = 0.
	void replaceBed(std::vector<float> zb);

  private:
	void applyParameters();
	void applyChronologies();
	void applyBridges();
	bool validMask(CellMask const &cells) const { return cells.size() == case_.cells(); }
	CaseData::Chronology &chronology(ChronologyKind kind);
	void updateBedRange();

	Solver                  solver_;
	CaseData                case_;
	std::pair<float, float> bedRange_{0.0f, 0.0f};
	GwStats                 gwStats_;
	SoilParams              soilDefaults_;
};

}        // namespace swe
