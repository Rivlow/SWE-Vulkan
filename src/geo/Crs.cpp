#include "geo/Crs.h"

#include "util/Log.h"

#include <cpl_conv.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <mutex>
#include <numbers>
#include <regex>

#ifdef _WIN32
#	include <windows.h>
#endif

namespace swe::geo {

namespace {

constexpr double kEarthRadius = 6378137.0;        // WGS 84 semi-major axis, used by Web Mercator
constexpr double kMaxLat      = 85.05112878;

std::filesystem::path exeDir()
{
#ifdef _WIN32
	wchar_t exe[MAX_PATH];
	if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0)
		return std::filesystem::path(exe).parent_path();
#endif
	return std::filesystem::current_path();
}

// Cached transformations between EPSG codes (OGR objects are not cheap to build). One cache per thread:
// an OGRCoordinateTransformation must not be used concurrently (PROJ contexts are per thread), and the
// download workers reproject in parallel.
class Transforms
{
  public:
	Transforms() = default;
	Transforms &operator=(Transforms &&) noexcept
	{
		cache_.clear();
		return *this;
	}

	OGRCoordinateTransformation *get(int from, int to)
	{
		auto key = std::make_pair(from, to);
		auto            it  = cache_.find(key);
		if (it != cache_.end())
			return it->second.get();
		OGRSpatialReference src, dst;
		if (src.importFromEPSG(from) != OGRERR_NONE || dst.importFromEPSG(to) != OGRERR_NONE)
		{
			log::error("EPSG:{} -> EPSG:{}: unknown CRS (is PROJ data available?)", from, to);
			return nullptr;
		}
		// x = easting / longitude, y = northing / latitude, whatever the authority says
		src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
		dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
		std::unique_ptr<OGRCoordinateTransformation> ct(OGRCreateCoordinateTransformation(&src, &dst));
		auto                                        *raw = ct.get();
		cache_[key]                                      = std::move(ct);
		return raw;
	}

  private:
	std::map<std::pair<int, int>, std::unique_ptr<OGRCoordinateTransformation>> cache_;
};

Transforms &transforms()
{
	thread_local Transforms t;
	return t;
}

}        // namespace

BBox BBox::intersect(BBox const &o) const
{
	return {std::max(xmin, o.xmin), std::max(ymin, o.ymin), std::min(xmax, o.xmax), std::min(ymax, o.ymax)};
}

void init()
{
	static std::once_flag once;
	std::call_once(once, [] {
		// resource files: next to the executable (packaged app) or in the vcpkg install (development)
		std::filesystem::path exe = exeDir();
		std::string           proj = (exe / "proj").string();
		std::string           gdal = (exe / "gdal").string();
		if (!std::filesystem::exists(proj + "/proj.db"))
			proj = SWE_PROJ_DATA;
		if (!std::filesystem::exists(gdal))
			gdal = SWE_GDAL_DATA;
		char const *paths[] = {proj.c_str(), nullptr};
		OSRSetPROJSearchPaths(paths);
		CPLSetConfigOption("GDAL_DATA", gdal.c_str());
		// quiet, no exceptions: errors are checked by return values
		CPLSetErrorHandler(CPLQuietErrorHandler);
		// HTTP (vsicurl, WCS): identify the application, keep going on slow servers
		CPLSetConfigOption("GDAL_HTTP_USERAGENT", "SWE-Vulkan/0.2 (shallow-water simulator; contact via GitHub)");
		CPLSetConfigOption("GDAL_HTTP_TIMEOUT", "120");
		CPLSetConfigOption("GDAL_HTTP_MAX_RETRY", "3");
		CPLSetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR");
		CPLSetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff,.zip,.json,.txt,.xml");
		GDALAllRegister();
		if (!std::filesystem::exists(proj + "/proj.db"))
			log::warn("proj.db not found in {}: coordinate transformations will fail", proj);
		else
			log::info("GDAL {} ready (PROJ data: {})", GDALVersionInfo("RELEASE_NAME"), proj);
	});
}

void shutdown()
{
	transforms() = Transforms{};
	GDALDestroyDriverManager();
	OSRCleanup();
	CPLCleanupTLS();
}

bool transform(int fromEpsg, int toEpsg, double &x, double &y)
{
	if (fromEpsg == toEpsg)
		return true;
	init();
	auto *ct = transforms().get(fromEpsg, toEpsg);
	return ct && ct->Transform(1, &x, &y) != 0;
}

LonLat toLonLat(XY p)
{
	double x = p.x, y = p.y;
	transform(EPSG_LAMBERT72, EPSG_WGS84, x, y);
	return {x, y};
}

XY toLambert(LonLat ll)
{
	double x = ll.lon, y = ll.lat;
	transform(EPSG_WGS84, EPSG_LAMBERT72, x, y);
	return {x, y};
}

XY mercator(LonLat ll)
{
	double lat = std::clamp(ll.lat, -kMaxLat, kMaxLat) * std::numbers::pi / 180.0;
	return {kEarthRadius * ll.lon * std::numbers::pi / 180.0, kEarthRadius * std::log(std::tan(std::numbers::pi / 4 + lat / 2))};
}

LonLat fromMercator(XY m)
{
	double lon = m.x / kEarthRadius * 180.0 / std::numbers::pi;
	double lat = (2.0 * std::atan(std::exp(m.y / kEarthRadius)) - std::numbers::pi / 2) * 180.0 / std::numbers::pi;
	return {lon, lat};
}

TileXY toTile(LonLat ll, int z)
{
	double n   = std::ldexp(1.0, z);
	double lat = std::clamp(ll.lat, -kMaxLat, kMaxLat) * std::numbers::pi / 180.0;
	return {(ll.lon + 180.0) / 360.0 * n, (1.0 - std::log(std::tan(lat) + 1.0 / std::cos(lat)) / std::numbers::pi) / 2.0 * n};
}

LonLat fromTile(TileXY t, int z)
{
	double n   = std::ldexp(1.0, z);
	double lon = t.x / n * 360.0 - 180.0;
	double lat = std::atan(std::sinh(std::numbers::pi * (1.0 - 2.0 * t.y / n))) * 180.0 / std::numbers::pi;
	return {lon, lat};
}

namespace {

template <class F>
BBox hull(BBox const &b, F const &f)
{
	BBox   out{1e300, 1e300, -1e300, -1e300};
	double xs[3] = {b.xmin, 0.5 * (b.xmin + b.xmax), b.xmax};
	double ys[3] = {b.ymin, 0.5 * (b.ymin + b.ymax), b.ymax};
	for (double x : xs)
		for (double y : ys)
		{
			auto p   = f(x, y);
			out.xmin = std::min(out.xmin, p.x);
			out.xmax = std::max(out.xmax, p.x);
			out.ymin = std::min(out.ymin, p.y);
			out.ymax = std::max(out.ymax, p.y);
		}
	return out;
}

}        // namespace

BBox transformBox(BBox const &box, int fromEpsg, int toEpsg)
{
	if (fromEpsg == toEpsg)
		return box;
	return hull(box, [&](double x, double y) {
		double px = x, py = y;
		transform(fromEpsg, toEpsg, px, py);
		return XY{px, py};
	});
}

int suggestEpsg(LonLat c)
{
	// Belgium and its immediate surroundings: the national grid (and the SPW / Flemish data are native in it)
	if (c.lon >= 2.3 && c.lon <= 6.6 && c.lat >= 49.3 && c.lat <= 51.7)
		return EPSG_LAMBERT72;
	int zone = static_cast<int>(std::floor((c.lon + 180.0) / 6.0)) + 1;
	zone     = std::clamp(zone, 1, 60);
	return (c.lat >= 0.0 ? 32600 : 32700) + zone;
}

std::string crsName(int epsg)
{
	if (epsg == EPSG_LAMBERT72)
		return "Lambert 72";
	if (epsg > 32600 && epsg <= 32660)
		return "UTM " + std::to_string(epsg - 32600) + "N";
	if (epsg > 32700 && epsg <= 32760)
		return "UTM " + std::to_string(epsg - 32700) + "S";
	return "EPSG:" + std::to_string(epsg);
}

BBox lambertToLonLatBox(BBox const &l72)
{
	return hull(l72, [](double x, double y) {
		auto ll = toLonLat({x, y});
		return XY{ll.lon, ll.lat};
	});
}

BBox lonLatToLambertBox(BBox const &ll)
{
	return hull(ll, [](double lon, double lat) { return toLambert({lon, lat}); });
}

bool parseLocation(std::string const &text, LonLat &out)
{
	static std::regex const patterns[] = {
	    std::regex(R"(@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?))"),                       // google: /@50.63,5.57,15z
	    std::regex(R"([?&](?:q|ll|query|center)=(-?\d+(?:\.\d+)?)(?:,|%2C)(-?\d+(?:\.\d+)?))"),
	    std::regex(R"(#map=\d+/(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?))"),              // openstreetmap: #map=15/50.63/5.57
	    std::regex(R"(^\s*(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)\s*$)"),      // "50.63, 5.57"
	};
	for (auto const &re : patterns)
	{
		std::smatch m;
		if (std::regex_search(text, m, re))
		{
			double a = std::stod(m[1]), b = std::stod(m[2]);
			// Lambert 72 coordinates pasted directly (metres)
			if (std::abs(a) > 360.0 || std::abs(b) > 360.0)
			{
				out = toLonLat({a, b});
				return std::abs(out.lat) <= 90.0;
			}
			out = {b, a};        // lat, lon order in URLs and common usage
			return std::abs(out.lat) <= 90.0 && std::abs(out.lon) <= 180.0;
		}
	}
	return false;
}

}        // namespace swe::geo
