#pragma once
// An elevation (or bathymetry) data source that can fill a metric grid in any working projection.
#include "geo/Progress.h"
#include "geo/Raster.h"

#include <memory>
#include <string>
#include <vector>

namespace swe::geo {

enum class SourceKind
{
	Terrain,            // DTM: ground (water surfaces = water level or interpolation)
	Bathymetry,         // river / lake bed surveys, NoData elsewhere
};

struct SourceInfo
{
	std::string id;               // short stable identifier ("spw_bathy", "dhmv2", "ahn4", "nrw_dgm1", "ign_rge", "cop30")
	std::string name;             // for the GUI
	std::string provider;
	std::string licence;          // short licence name
	std::string attribution;      // text to show / to write in the exported case
	SourceKind  kind = SourceKind::Terrain;
	double      resolution = 1.0; // native cell (m)
	float       verticalOffset = 0.0f;   // added to the values to express them in DNG (Belgian datum), approximate
	bool        online = false;
	BBox        extentLonLat;     // approximate footprint (WGS 84 lon/lat) for the coverage estimate
};

class RasterSource
{
  public:
	virtual ~RasterSource() = default;
	virtual SourceInfo const &info() const = 0;

	// Cheap estimate: fraction of `box` (in `epsg`) inside the source footprint (0..1).
	double footprintCoverage(BBox const &box, int epsg) const;

	// Fills a grid of `cell` metres over `box` (in `epsg`), NaN where the source has no data. May be slow.
	// Returns an empty grid on failure and sets `error`.
	virtual Grid fetch(BBox const &box, double cell, int epsg, Progress &progress, std::string *error) = 0;
};

}        // namespace swe::geo
