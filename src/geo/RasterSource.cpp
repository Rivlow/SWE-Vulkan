#include "geo/RasterSource.h"

#include <algorithm>

namespace swe::geo {

double RasterSource::footprintCoverage(BBox const &box, int epsg) const
{
	auto const &e = info().extentLonLat;
	if (e.empty() || box.empty())
		return 0.0;
	BBox ll = transformBox(box, epsg, EPSG_WGS84);
	auto i  = ll.intersect(e);
	if (i.empty())
		return 0.0;
	return std::clamp(i.width() * i.height() / (ll.width() * ll.height()), 0.0, 1.0);
}

}        // namespace swe::geo
