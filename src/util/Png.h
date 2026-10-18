#pragma once
#include <cstdint>
#include <filesystem>

namespace swe {

// Writes 8-bit RGBA (or BGRA, swizzled) pixels as an opaque PNG. Creates the parent directory.
bool writePng(std::filesystem::path const &path, uint8_t const *pixels, uint32_t width, uint32_t height, bool bgra);

}        // namespace swe
