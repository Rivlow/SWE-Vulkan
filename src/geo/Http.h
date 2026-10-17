#pragma once
// Minimal HTTP GET with a disk cache (libcurl). Used for WCS/WMS responses and map tiles.
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace swe::geo::http {

struct Response
{
	long                 status = 0;
	std::string          contentType;
	std::vector<uint8_t> body;
	bool                 fromCache = false;
	std::string          error;
	bool                 ok() const { return status >= 200 && status < 300 && error.empty(); }
};

// Cache directory (default: ./cache/http). Created on demand.
void                  setCacheDir(std::filesystem::path const &dir);
std::filesystem::path cacheDir();

// GET `url`. With `cache`, a successful response is stored on disk and served from there next time
// (tile policies allow caching; WCS data is immutable for a given request).
Response get(std::string const &url, bool cache = true, int timeoutSeconds = 120);

// Cache file path for a URL (exists only after a successful cached GET)
std::filesystem::path cachePath(std::string const &url);

std::string urlEncode(std::string const &s);

}        // namespace swe::geo::http
