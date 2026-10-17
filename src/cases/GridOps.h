#pragma once
// Port of the numpy helpers of cases/wolf_dtm.py. Arrays are (nx, ny) in C order (index i * ny + j).
#include <cstdint>
#include <vector>

namespace swe::grid {

// Refines by an integer factor r with bilinear interpolation between cell centres
// (computed in double, like numpy with a float64 weight, then rounded to f32).
std::vector<float> refineBilinear(std::vector<float> const &a, uint32_t nx, uint32_t ny, uint32_t r);

// Refines by an integer factor r by repeating each cell (masks, basin numbers).
template <class T>
std::vector<T> refineNearest(std::vector<T> const &a, uint32_t nx, uint32_t ny, uint32_t r)
{
	std::vector<T> out(size_t(nx) * r * ny * r);
	uint32_t const nyr = ny * r;
	for (uint32_t i = 0; i < nx * r; ++i)
		for (uint32_t j = 0; j < nyr; ++j)
			out[size_t(i) * nyr + j] = a[size_t(i / r) * ny + j / r];
	return out;
}

// Cells outside the domain (NaN) -> walls: elevation of the nearest valid cell + wall_height
// (iterative 4-neighbour max dilation of the border, as the Python version).
std::vector<float> fillOutside(std::vector<float> const &z, uint32_t nx, uint32_t ny, float wallHeight);

// NaN-ignoring minimum and first argmin in C order (np.nanmin / np.nanargmin). Throws if all NaN.
float  nanMin(std::vector<float> const &a);
size_t nanArgMin(std::vector<float> const &a);

}        // namespace swe::grid
