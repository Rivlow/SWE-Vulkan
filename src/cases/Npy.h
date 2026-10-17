#pragma once
// Minimal .npy / .npz support (little-endian, C order). Enough for the cached data files and dumps.
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace swe::npy {

struct Array
{
	std::string          descr;        // e.g. "<f4", "<f8", "<i2", "|u1", "|b1"
	std::vector<size_t>  shape;
	std::vector<uint8_t> data;         // raw bytes, C order

	size_t size() const;               // number of elements
	size_t itemSize() const;

	// Element-wise conversion of any supported numeric dtype.
	std::vector<float>  asFloat() const;
	std::vector<double> asDouble() const;
};

Array load(std::filesystem::path const &path);
Array parse(uint8_t const *bytes, size_t length, std::string const &what);

void save(std::filesystem::path const &path, std::string const &descr, std::vector<size_t> const &shape, void const *data,
          size_t bytes);

inline void save(std::filesystem::path const &path, std::vector<size_t> const &shape, std::vector<float> const &v)
{
	save(path, "<f4", shape, v.data(), v.size() * sizeof(float));
}

// .npz archives written by np.savez (stored, not compressed).
std::map<std::string, Array> loadNpz(std::filesystem::path const &path);

// Writes a stored (uncompressed) .npz readable by np.load. Each entry: name -> array.
void saveNpz(std::filesystem::path const &path, std::map<std::string, Array> const &arrays);

// Helpers to build Arrays from host vectors
Array makeArray(std::vector<float> const &v, std::vector<size_t> const &shape);
Array makeArray(std::vector<uint8_t> const &v, std::vector<size_t> const &shape);
Array makeArray(std::vector<double> const &v, std::vector<size_t> const &shape);

}        // namespace swe::npy
