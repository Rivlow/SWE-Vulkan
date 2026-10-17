#include "geo/RasterSource.h"

#include <algorithm>

namespace swe::geo {

double RasterSource::footprintCoverage(BBox const &box) const
{
	auto const &e = info().extentL72;
	if (e.empty() || box.empty())
		return 0.0;
	auto   i = box.intersect(e);
	if (i.empty())
		return 0.0;
	return std::clamp(i.width() * i.height() / (box.width() * box.height()), 0.0, 1.0);
}

}        // namespace swe::geo
