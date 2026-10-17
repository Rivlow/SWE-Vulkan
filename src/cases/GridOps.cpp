#include "cases/GridOps.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace swe::grid {

namespace {

// Interpolation coordinates of numpy's refine_bilinear along one axis of length n
struct Coord
{
	uint32_t i0, i1;
	double   t;
};

std::vector<Coord> coords(uint32_t n, uint32_t r)
{
	std::vector<Coord> c(size_t(n) * r);
	for (uint32_t k = 0; k < n * r; ++k)
	{
		double   x  = std::clamp((k + 0.5) / r - 0.5, 0.0, double(n) - 1.0);        // fine centres in coarse indices
		uint32_t i0 = std::min(static_cast<uint32_t>(x), n >= 2 ? n - 2 : 0u);
		c[k]        = {i0, std::min(i0 + 1, n - 1), x - i0};
	}
	return c;
}

}        // namespace

std::vector<float> refineBilinear(std::vector<float> const &a, uint32_t nx, uint32_t ny, uint32_t r)
{
	auto cx = coords(nx, r), cy = coords(ny, r);
	// axis 0 (float32 input * float64 weight -> float64)
	std::vector<double> tmp(size_t(nx) * r * ny);
	for (uint32_t k = 0; k < nx * r; ++k)
		for (uint32_t j = 0; j < ny; ++j)
			tmp[size_t(k) * ny + j] = (1.0 - cx[k].t) * a[size_t(cx[k].i0) * ny + j] + cx[k].t * a[size_t(cx[k].i1) * ny + j];
	// axis 1, then astype(float32)
	uint32_t const     nyr = ny * r;
	std::vector<float> out(size_t(nx) * r * nyr);
	for (uint32_t k = 0; k < nx * r; ++k)
		for (uint32_t l = 0; l < nyr; ++l)
		{
			double v = (1.0 - cy[l].t) * tmp[size_t(k) * ny + cy[l].i0] + cy[l].t * tmp[size_t(k) * ny + cy[l].i1];
			out[size_t(k) * nyr + l] = static_cast<float>(v);
		}
	return out;
}

std::vector<float> fillOutside(std::vector<float> const &zIn, uint32_t nx, uint32_t ny, float wallHeight)
{
	constexpr float    ninf = -std::numeric_limits<float>::infinity();
	std::vector<float> z    = zIn;
	std::vector<char>  outside(z.size()), known(z.size());
	for (size_t k = 0; k < z.size(); ++k)
	{
		outside[k] = std::isnan(z[k]);
		known[k]   = !outside[k];
	}
	if (std::none_of(known.begin(), known.end(), [](char c) { return c; }))
		throw std::runtime_error("fill_outside: no valid cell");

	auto at = [&](std::vector<float> const &src, std::vector<char> const &kn, int i, int j) {
		if (i < 0 || j < 0 || i >= int(nx) || j >= int(ny))
			return ninf;
		size_t k = size_t(i) * ny + j;
		return kn[k] ? src[k] : ninf;
	};

	bool remaining = true;
	while (remaining)
	{
		// every pass reads the state at its start (numpy evaluates the whole array at once)
		std::vector<float> zPrev     = z;
		std::vector<char>  knownPrev = known;
		remaining                    = false;
		for (int i = 0; i < int(nx); ++i)
			for (int j = 0; j < int(ny); ++j)
			{
				size_t k = size_t(i) * ny + j;
				if (knownPrev[k])
					continue;
				float neigh = std::max({at(zPrev, knownPrev, i - 1, j), at(zPrev, knownPrev, i + 1, j),
				                        at(zPrev, knownPrev, i, j - 1), at(zPrev, knownPrev, i, j + 1)});
				if (std::isfinite(neigh))
				{
					z[k]     = neigh;
					known[k] = 1;
				}
				else
					remaining = true;
			}
	}
	for (size_t k = 0; k < z.size(); ++k)
		if (outside[k])
			z[k] += wallHeight;
	return z;
}

float nanMin(std::vector<float> const &a)
{
	return a[nanArgMin(a)];
}

size_t nanArgMin(std::vector<float> const &a)
{
	size_t best = a.size();
	for (size_t k = 0; k < a.size(); ++k)
		if (!std::isnan(a[k]) && (best == a.size() || a[k] < a[best]))
			best = k;
	if (best == a.size())
		throw std::runtime_error("nanargmin: all values are NaN");
	return best;
}

}        // namespace swe::grid
