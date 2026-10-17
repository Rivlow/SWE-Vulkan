#pragma once
// Composes a solver bed from the available sources: local DTM (Wallonia downloads) > online neighbours >
// Copernicus, then bathymetry on top, holes filled, and turns it into a CaseData.
#include "cases/Case.h"
#include "geo/LocalLibrary.h"
#include "geo/RasterSource.h"

#include <memory>
#include <string>
#include <vector>

namespace swe::geo {

struct SourceReport
{
	std::string id, name;
	double      footprint = 0.0;        // fraction of the box inside the source footprint
	double      data      = -1.0;       // fraction of cells with data after a (coarse) fetch, -1 = not fetched
	bool        used      = false;      // contributed cells to the final bed
	std::string error;
};

enum class InitialWater
{
	Dry,                // no water
	Level,              // still water at a given elevation everywhere below it
	LevelInRectangle,   // still water at a given elevation inside a Lambert 72 rectangle (reservoir, lake)
};

struct ImportRequest
{
	BBox         box;                    // in `epsg`
	int          epsg       = EPSG_LAMBERT72;   // working projection (see suggestEpsg)
	double       cell       = 5.0;       // target resolution (m)
	bool         mergeBathy = true;
	bool         fillHoles  = true;
	std::string  name       = "import";
	std::string  label;
	InitialWater initial    = InitialWater::Dry;
	double       level      = 0.0;       // absolute elevation (m DNG)
	BBox         levelBox;               // for LevelInRectangle
	double       manning    = 0.03;
	double       depthRef   = 3.0;
	double       zScale     = 3.0;
};

struct ImportResult
{
	Grid                      bed;         // absolute elevations (m DNG), holes filled
	std::vector<uint8_t>      outside;     // 1 = no data in any source (filled by dilation)
	std::vector<uint8_t>      bathyMask;   // 1 = bathymetry used
	std::vector<SourceReport> sources;
	std::vector<std::string>  attributions;
	double                    bathyCoverage = 0.0;
};

class TerrainBuilder
{
  public:
	explicit TerrainBuilder(LocalLibrary &library);

	LocalLibrary                                      &library() { return library_; }
	std::vector<std::unique_ptr<RasterSource>> const  &terrainSources() const { return terrain_; }
	RasterSource                                      &bathymetrySource() { return *bathy_; }
	void                                               refreshLocal();        // after a library rescan

	// Quick look at what covers `box`: footprints, plus a coarse fetch (~96 px across) for the data fraction.
	std::vector<SourceReport> probe(BBox const &box, int epsg, Progress &progress, bool fetchCoarse = true);

	// Full import. Empty bed on failure (error in `error`).
	ImportResult build(ImportRequest const &req, Progress &progress, std::string *error);

	// Converts an import into a case (initial conditions, boundaries = walls, outside mask).
	static CaseData toCase(ImportResult const &result, ImportRequest const &req);

  private:
	LocalLibrary                              &library_;
	std::vector<std::unique_ptr<RasterSource>> terrain_;        // priority order
	std::unique_ptr<RasterSource>              bathy_;
};

}        // namespace swe::geo
