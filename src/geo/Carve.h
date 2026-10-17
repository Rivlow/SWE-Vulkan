#pragma once
// Bed editing tools for areas without surveyed bathymetry: LiDAR DTMs show the water surface, not the bed.
//   carveChannel : lowers the bed along a polyline (river axis) with a parabolic cross-section
//   lowerPolygon : lowers (or raises) every cell inside a polygon by a constant
// Coordinates are Lambert 72 metres; the grid is the solver layout (index i * ny + j).
#include "geo/Crs.h"

#include <cstdint>
#include <vector>

namespace swe::geo {

struct GridGeometry
{
	uint32_t nx = 0, ny = 0;
	double   x0 = 0, y0 = 0, cell = 1.0;
	int      epsg = EPSG_LAMBERT72;        // projection of x0, y0 and of the points given to the tools
};

struct ChannelSpec
{
	std::vector<XY> axis;             // >= 2 points
	double          width = 20.0;     // top width (m)
	double          depth = 2.0;      // depth below the reference surface at the axis (m)
	// reference surface along the axis: the lowest bed elevation found within `width/2` of the axis point
	// (the water surface in a LiDAR DTM), smoothed along the axis. The carved bed never rises above the
	// existing bed.
	double          bankSmoothing = 1.0;   // 0 = sharp banks, 1 = parabola reaching the top width
};

// Returns the number of modified cells. `zb` holds relative elevations (any constant offset is fine).
size_t carveChannel(std::vector<float> &zb, GridGeometry const &g, ChannelSpec const &spec);

size_t lowerPolygon(std::vector<float> &zb, GridGeometry const &g, std::vector<XY> const &polygon, float dz);

bool pointInPolygon(std::vector<XY> const &polygon, double x, double y);

}        // namespace swe::geo
