#pragma once
// Where the application finds and writes its files. A packaged application (installer) lives in a per-user
// folder with everything next to the executable: assets/, shaders/, proj/, gdal/, swe_settings.json, captures/.
// A development build finds the same things next to the executable or falls back to the source tree.
#include <filesystem>

namespace swe::paths {

// Directory of the running executable
std::filesystem::path exeDir();
// Writable directory for the user's files (settings, captures, imported cases): next to the executable
std::filesystem::path dataDir();

}        // namespace swe::paths
