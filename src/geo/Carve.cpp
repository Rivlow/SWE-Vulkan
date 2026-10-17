#include "geo/Carve.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace swe::geo {

namespace {

// distance from p to segment ab, and the parameter t in [0, 1] of the closest point
double segmentDistance(XY p, XY a, XY b, double &t)
{
	double dx = b.x - a.x, dy = b.y - a.y;
	double l2 = dx * dx + dy * dy;
	t         = l2 > 0 ? std::clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / l2, 0.0, 1.0) : 0.0;
	double cx = a.x + t * dx - p.x, cy = a.y + t * dy - p.y;
	return std::sqrt(cx * cx + cy * cy);
}

}        // namespace

bool pointInPolygon(std::vector<XY> const &poly, double x, double y)
{
	bool inside = false;
	for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
	{
		bool cross = (poly[i].y > y) != (poly[j].y > y);
		if (cross && x < (poly[j].x - poly[i].x) * (y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)
			inside = !inside;
	}
	return inside;
}

size_t carveChannel(std::vector<float> &zb, GridGeometry const &g, ChannelSpec const &spec)
{
	if (spec.axis.size() < 2 || spec.width <= 0.0 || spec.depth <= 0.0 || zb.size() != size_t(g.nx) * g.ny)
		return 0;
	double half = 0.5 * spec.width;

	// 1. arc length along the axis and the reference surface at each vertex: lowest bed within half width
	size_t              n = spec.axis.size();
	std::vector<double> s(n, 0.0), ref(n);
	for (size_t k = 1; k < n; ++k)
		s[k] = s[k - 1] + std::hypot(spec.axis[k].x - spec.axis[k - 1].x, spec.axis[k].y - spec.axis[k - 1].y);
	auto bedAt = [&](double x, double y) -> float {
		long i = std::lround((x - g.x0) / g.cell - 0.5), j = std::lround((y - g.y0) / g.cell - 0.5);
		if (i < 0 || j < 0 || i >= long(g.nx) || j >= long(g.ny))
			return std::numeric_limits<float>::quiet_NaN();
		return zb[size_t(i) * g.ny + size_t(j)];
	};
	for (size_t k = 0; k < n; ++k)
	{
		float  lowest = std::numeric_limits<float>::infinity();
		int    r      = std::max(1, int(std::ceil(half / g.cell)));
		for (int a = -r; a <= r; ++a)
			for (int b = -r; b <= r; ++b)
			{
				if (double(a * a + b * b) * g.cell * g.cell > half * half)
					continue;
				float z = bedAt(spec.axis[k].x + a * g.cell, spec.axis[k].y + b * g.cell);
				if (std::isfinite(z))
					lowest = std::min(lowest, z);
			}
		ref[k] = std::isfinite(lowest) ? lowest : (k > 0 ? ref[k - 1] : 0.0);
	}
	// smooth the reference along the axis (3-point) so the bed does not jump between vertices
	for (int pass = 0; pass < 2; ++pass)
	{
		auto prev = ref;
		for (size_t k = 1; k + 1 < n; ++k)
			ref[k] = 0.25 * prev[k - 1] + 0.5 * prev[k] + 0.25 * prev[k + 1];
	}

	// 2. cells within half width of the polyline: parabolic deepening below the interpolated reference
	size_t modified = 0;
	BBox   bb{1e300, 1e300, -1e300, -1e300};
	for (auto const &p : spec.axis)
	{
		bb.xmin = std::min(bb.xmin, p.x - half);
		bb.xmax = std::max(bb.xmax, p.x + half);
		bb.ymin = std::min(bb.ymin, p.y - half);
		bb.ymax = std::max(bb.ymax, p.y + half);
	}
	long i0 = std::max(0L, long(std::floor((bb.xmin - g.x0) / g.cell))), i1 = std::min(long(g.nx) - 1, long(std::ceil((bb.xmax - g.x0) / g.cell)));
	long j0 = std::max(0L, long(std::floor((bb.ymin - g.y0) / g.cell))), j1 = std::min(long(g.ny) - 1, long(std::ceil((bb.ymax - g.y0) / g.cell)));
	for (long i = i0; i <= i1; ++i)
		for (long j = j0; j <= j1; ++j)
		{
			XY     p{g.x0 + (i + 0.5) * g.cell, g.y0 + (j + 0.5) * g.cell};
			double best = std::numeric_limits<double>::infinity(), bestS = 0.0;
			for (size_t k = 0; k + 1 < n; ++k)
			{
				double t, d = segmentDistance(p, spec.axis[k], spec.axis[k + 1], t);
				if (d < best)
				{
					best  = d;
					bestS = s[k] + t * (s[k + 1] - s[k]);
				}
			}
			if (best > half)
				continue;
			// reference surface interpolated along the axis
			size_t k = 0;
			while (k + 2 < n && s[k + 1] < bestS)
				++k;
			double t    = s[k + 1] > s[k] ? (bestS - s[k]) / (s[k + 1] - s[k]) : 0.0;
			double zref = ref[k] + t * (ref[k + 1] - ref[k]);
			double u    = best / half;        // 0 at the axis, 1 at the bank
			double shape = spec.bankSmoothing * (1.0 - u * u) + (1.0 - spec.bankSmoothing) * (u < 0.85 ? 1.0 : (1.0 - u) / 0.15);
			float  target = static_cast<float>(zref - spec.depth * std::clamp(shape, 0.0, 1.0));
			auto  &z      = zb[size_t(i) * g.ny + size_t(j)];
			if (target < z)
			{
				z = target;
				++modified;
			}
		}
	return modified;
}

size_t lowerPolygon(std::vector<float> &zb, GridGeometry const &g, std::vector<XY> const &polygon, float dz)
{
	if (polygon.size() < 3 || zb.size() != size_t(g.nx) * g.ny)
		return 0;
	BBox bb{1e300, 1e300, -1e300, -1e300};
	for (auto const &p : polygon)
	{
		bb.xmin = std::min(bb.xmin, p.x);
		bb.xmax = std::max(bb.xmax, p.x);
		bb.ymin = std::min(bb.ymin, p.y);
		bb.ymax = std::max(bb.ymax, p.y);
	}
	long i0 = std::max(0L, long(std::floor((bb.xmin - g.x0) / g.cell))), i1 = std::min(long(g.nx) - 1, long(std::ceil((bb.xmax - g.x0) / g.cell)));
	long j0 = std::max(0L, long(std::floor((bb.ymin - g.y0) / g.cell))), j1 = std::min(long(g.ny) - 1, long(std::ceil((bb.ymax - g.y0) / g.cell)));
	size_t modified = 0;
	for (long i = i0; i <= i1; ++i)
		for (long j = j0; j <= j1; ++j)
			if (pointInPolygon(polygon, g.x0 + (i + 0.5) * g.cell, g.y0 + (j + 0.5) * g.cell))
			{
				zb[size_t(i) * g.ny + size_t(j)] -= dz;
				++modified;
			}
	return modified;
}

}        // namespace swe::geo
