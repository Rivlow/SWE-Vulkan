#pragma once
// Regular float grids in Lambert 72, in the solver's layout: index i * ny + j, i along x (east), j along y
// (north), cell (i, j) centred at (x0 + (i + 0.5) cell, y0 + (j + 0.5) cell). GDAL stores rows from the top:
// fromGdalRows() / toGdalRows() convert.
#include "geo/Crs.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

class GDALDataset;

namespace swe::geo {

struct Grid
{
	uint32_t           nx = 0, ny = 0;
	double             x0 = 0, y0 = 0;        // lower-left corner (m, Lambert 72)
	double             cell = 1.0;            // square cells
	float              nodata = std::numeric_limits<float>::quiet_NaN();
	std::vector<float> v;

	Grid() = default;
	Grid(BBox const &box, double cellSize, float nodataValue = std::numeric_limits<float>::quiet_NaN());

	size_t cells() const { return size_t(nx) * ny; }
	BBox   bounds() const { return {x0, y0, x0 + nx * cell, y0 + ny * cell}; }
	float &at(uint32_t i, uint32_t j) { return v[size_t(i) * ny + j]; }
	float  at(uint32_t i, uint32_t j) const { return v[size_t(i) * ny + j]; }
	bool   isNoData(float f) const;
	size_t countNoData() const;
	// fraction of cells holding data
	double coverage() const;

	// Fill this grid's NoData cells with src values (src resampled to this grid beforehand: same geometry).
	void fillFrom(Grid const &src);
	// Replace cells where src has data (bathymetry over a DTM).
	void overrideFrom(Grid const &src);
	// Add a constant to every data cell (vertical datum offsets).
	void shift(float dz);
	// Grow data into NoData cells by nearest-neighbour dilation (max of 4 neighbours like fill_outside), then
	// returns the mask of cells that were NoData before (1 = filled).
	std::vector<uint8_t> fillHoles();

	// row-major top-down (GDAL) <-> solver layout
	static Grid fromGdalRows(std::vector<float> const &rows, uint32_t width, uint32_t height, BBox const &box,
	                         double cellSize, float nodataValue);
	std::vector<float> toGdalRows() const;
};

// Metadata of a GDAL raster (any CRS).
struct RasterInfo
{
	std::string path;          // GDAL path (may be /vsizip/... or /vsicurl/...)
	int         epsg = 0;      // 0 = unknown / not an EPSG code
	BBox        bounds;        // in the raster's CRS
	BBox        boundsL72;     // in Lambert 72
	double      resX = 0, resY = 0;        // in the raster's CRS units (degrees for geographic rasters)
	double      resMetres = 0;             // approximate ground resolution (m)
	uint32_t    width = 0, height = 0;
	float       nodata = std::numeric_limits<float>::quiet_NaN();
	bool        hasNoData = false;
	int         bands = 0;
	std::string dataType;
	std::string description;
};

bool readInfo(std::string const &path, RasterInfo &info, std::string *error = nullptr);

// Reads the raster window covering `box` (Lambert 72) and resamples it onto a grid of `cell` metres:
// GDALWarp to EPSG:31370 (bilinear when refining, average when coarsening), NoData preserved.
// Returns an empty grid (nx = 0) on failure.
Grid readWindow(std::string const &path, BBox const &box, double cell, std::string *error = nullptr);

// Writes a grid as a Lambert 72 GeoTIFF (float32, deflate) - debugging / export.
bool writeGeoTiff(std::string const &path, Grid const &grid, std::string *error = nullptr);

}        // namespace swe::geo
