#include "geo/OnlineSources.h"

#include "geo/Http.h"
#include "util/Log.h"

#include <cpl_conv.h>
#include <cpl_vsi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <mutex>
#include <thread>

namespace swe::geo {

namespace {

// Writes bytes into GDAL's in-memory file system and reads them as a raster window.
Grid gridFromBytes(std::vector<uint8_t> const &bytes, std::string const &name, BBox const &box, double cell, int epsg, int srcEpsg,
                   std::string *error)
{
	std::string vsi = "/vsimem/swe_" + name + ".tif";
	// VSIFileFromMemBuffer takes ownership only when bTakeOwnership = TRUE; keep a copy GDAL may free
	auto *copy = static_cast<GByte *>(CPLMalloc(bytes.size()));
	std::copy(bytes.begin(), bytes.end(), copy);
	VSILFILE *f = VSIFileFromMemBuffer(vsi.c_str(), copy, static_cast<vsi_l_offset>(bytes.size()), TRUE);
	if (!f)
	{
		if (error)
			*error = "vsimem failure";
		return {};
	}
	VSIFCloseL(f);
	Grid g = readWindow(vsi, box, cell, epsg, error, srcEpsg);
	VSIUnlink(vsi.c_str());
	return g;
}

bool looksLikeTiff(std::vector<uint8_t> const &b)
{
	return b.size() > 4 && ((b[0] == 'I' && b[1] == 'I' && b[2] == 42) || (b[0] == 'M' && b[1] == 'M' && b[3] == 42));
}

// A WCS 2.0.1 server may wrap the GeoTIFF in a multipart/related body: find the TIFF header.
std::vector<uint8_t> extractTiff(std::vector<uint8_t> const &body)
{
	if (looksLikeTiff(body))
		return body;
	for (size_t k = 0; k + 4 < body.size(); ++k)
		if ((body[k] == 'I' && body[k + 1] == 'I' && body[k + 2] == 42 && body[k + 3] == 0) ||
		    (body[k] == 'M' && body[k + 1] == 'M' && body[k + 2] == 0 && body[k + 3] == 42))
			return {body.begin() + static_cast<std::ptrdiff_t>(k), body.end()};
	return {};
}

std::string fmt(double v)
{
	return std::format("{:.3f}", v);
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// OGC WCS / WMS
// ---------------------------------------------------------------------------------------------

OgcSource::OgcSource(SourceInfo info, OgcService service) :
    info_(std::move(info)), service_(std::move(service))
{
	info_.online = true;
}

std::string OgcSource::requestUrl(BBox const &b, uint32_t w, uint32_t h, bool outputL72) const
{
	auto const &s   = service_;
	std::string sep = s.baseUrl.find('?') == std::string::npos ? "?" : "&";
	switch (s.protocol)
	{
	case OgcService::Protocol::Wcs100:
		return std::format("{}{}SERVICE=WCS&VERSION=1.0.0&REQUEST=GetCoverage&COVERAGE={}&CRS=EPSG:{}&BBOX={},{},{},{}"
		                   "&WIDTH={}&HEIGHT={}&FORMAT=GeoTIFF",
		                   s.baseUrl, sep, s.layer, s.requestEpsg, fmt(b.xmin), fmt(b.ymin), fmt(b.xmax), fmt(b.ymax), w, h);
	case OgcService::Protocol::Wcs201:
	{
		std::string url = std::format("{}{}SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={}"
		                              "&SUBSET=x({},{})&SUBSET=y({},{})&FORMAT=image/tiff",
		                              s.baseUrl, sep, s.layer, fmt(b.xmin), fmt(b.xmax), fmt(b.ymin), fmt(b.ymax));
		std::string crs = std::format("http://www.opengis.net/def/crs/EPSG/0/{}", s.requestEpsg);
		url += "&SUBSETTINGCRS=" + http::urlEncode(crs);
		if (outputL72)
			url += "&OUTPUTCRS=" + http::urlEncode("http://www.opengis.net/def/crs/EPSG/0/31370");
		return url;
	}
	case OgcService::Protocol::WmsGeoTiff:
		return std::format("{}{}SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={}&STYLES=&CRS=EPSG:{}&BBOX={},{},{},{}"
		                   "&WIDTH={}&HEIGHT={}&FORMAT=image/geotiff",
		                   s.baseUrl, sep, s.layer, s.requestEpsg, fmt(b.xmin), fmt(b.ymin), fmt(b.xmax), fmt(b.ymax), w, h);
	}
	return {};
}

namespace {

// Runs `job(k)` for k in [0, n) on at most `workers` threads (I/O bound: several HTTP requests in flight).
template <class Job>
void parallelFor(uint32_t n, unsigned workers, Job const &job)
{
	std::atomic<uint32_t>      next{0};
	std::vector<std::jthread>  threads;
	unsigned                   count = std::min<unsigned>(std::max(1u, workers), std::max(1u, n));
	for (unsigned t = 0; t < count; ++t)
		threads.emplace_back([&] {
			for (uint32_t k = next.fetch_add(1); k < n; k = next.fetch_add(1))
				job(k);
		});
}

constexpr unsigned kParallelRequests = 6;

}        // namespace

Grid OgcSource::fetch(BBox const &box, double cell, int epsg, Progress &progress, std::string *error)
{
	Grid out(box, cell, epsg);
	// request resolution: never finer than the native product; twice the target resolution is enough for
	// the "average" warp onto the grid (cell 5 m -> 2.5 m requested instead of 1 m: 6x fewer bytes)
	double reqRes = std::max(service_.nativeRes, cell / 2.0);
	// request box in the server's CRS (hull of the target box) with a margin for the warp
	BBox   reqBox = transformBox(box, epsg, service_.requestEpsg);
	double m      = 2.0 * reqRes;
	reqBox        = {reqBox.xmin - m, reqBox.ymin - m, reqBox.xmax + m, reqBox.ymax + m};
	// the server reprojects for us only when the target is Lambert 72 and it supports it
	bool outputL72 = service_.outputL72 && epsg == EPSG_LAMBERT72;

	// WCS 2.0.1: the scaling extension (SCALEFACTOR) gives the requested resolution; without it (or when the
	// server rejects it, remembered in scaling_) the answer is at native resolution and tiles must be smaller
	bool wcs2    = service_.protocol == OgcService::Protocol::Wcs201;
	bool scaled  = !wcs2 || scaling_.load() != Scaling::Unsupported;
	if (wcs2 && !scaled)
		reqRes = service_.nativeRes;
	// tiles of at most 1024 px: several moderate requests in parallel finish sooner than one large one
	double   tileM = std::min(service_.maxPixels, 1024) * reqRes * 0.98;
	uint32_t tx    = static_cast<uint32_t>(std::ceil(reqBox.width() / tileM));
	uint32_t ty    = static_cast<uint32_t>(std::ceil(reqBox.height() / tileM));
	uint32_t total = tx * ty;

	std::mutex            mutex;        // out, error
	std::atomic<uint32_t> done{0};
	std::atomic<bool>     any{false};
	int                   srcEpsg = outputL72 ? EPSG_LAMBERT72 : service_.requestEpsg;

	auto fetchTile = [&](BBox const &t, uint32_t w, uint32_t h, double res, bool useScaling, uint32_t k) -> bool {
		auto url = requestUrl(t, w, h, outputL72);
		if (wcs2 && useScaling)
			url += std::format("&SCALEFACTOR={:.6f}", service_.nativeRes / res);
		auto r = http::get(url);
		if (!r.ok())
		{
			std::lock_guard lock(mutex);
			if (error)
				*error = r.error.empty() ? std::format("HTTP {}", r.status) : r.error;
			return false;
		}
		auto tiff = extractTiff(r.body);
		if (tiff.empty())
		{
			std::string head(r.body.begin(), r.body.begin() + std::min<size_t>(r.body.size(), 300));
			if (wcs2 && useScaling)
			{
				// the server does not implement the scaling extension: native tiles from now on
				if (scaling_.exchange(Scaling::Unsupported) != Scaling::Unsupported)
					log::info("{}: pas de mise à l'échelle côté serveur, tuiles natives", info_.name);
				return false;
			}
			log::warn("{}: réponse non TIFF: {}", info_.name, head);
			std::lock_guard lock(mutex);
			if (error)
				*error = "réponse du serveur non exploitable";
			return false;
		}
		if (wcs2 && useScaling)
			scaling_.store(Scaling::Supported);
		std::string err;
		Grid        part = gridFromBytes(tiff, std::format("{}_{}", info_.id, k), box, cell, epsg, srcEpsg, &err);
		if (part.v.empty())
		{
			log::warn("{}: {}", info_.name, err);
			return false;
		}
		std::lock_guard lock(mutex);
		out.fillFrom(part);
		any.store(true);
		return true;
	};

	auto tileBox = [&](uint32_t k, double tm) {
		uint32_t a = k % tx, b = k / tx;
		return BBox{reqBox.xmin + a * tm, reqBox.ymin + b * tm, std::min(reqBox.xmin + (a + 1) * tm, reqBox.xmax),
		            std::min(reqBox.ymin + (b + 1) * tm, reqBox.ymax)};
	};

	parallelFor(total, kParallelRequests, [&](uint32_t k) {
		if (progress.cancelled())
			return;
		BBox t = tileBox(k, tileM);
		auto w = static_cast<uint32_t>(std::max(1.0, std::round(t.width() / reqRes)));
		auto h = static_cast<uint32_t>(std::max(1.0, std::round(t.height() / reqRes)));
		fetchTile(t, w, h, reqRes, scaled && wcs2, k);
		uint32_t d = ++done;
		progress.set(float(d) / float(total), std::format("{} : tuile {}/{}", info_.name, d, total));
	});

	// scaling refused by a WCS 2.0.1 server: redo the whole area with native tiles (same code path, no scaling)
	if (wcs2 && scaled && scaling_.load() == Scaling::Unsupported && !progress.cancelled())
	{
		double   nres = service_.nativeRes, ntile = service_.maxPixels * nres * 0.98;
		uint32_t ntx = static_cast<uint32_t>(std::ceil(reqBox.width() / ntile)), nty = static_cast<uint32_t>(std::ceil(reqBox.height() / ntile));
		uint32_t ntotal = ntx * nty;
		done.store(0);
		parallelFor(ntotal, kParallelRequests, [&](uint32_t k) {
			if (progress.cancelled())
				return;
			uint32_t a = k % ntx, b = k / ntx;
			BBox t{reqBox.xmin + a * ntile, reqBox.ymin + b * ntile, std::min(reqBox.xmin + (a + 1) * ntile, reqBox.xmax),
			       std::min(reqBox.ymin + (b + 1) * ntile, reqBox.ymax)};
			auto w = static_cast<uint32_t>(std::max(1.0, std::round(t.width() / nres)));
			auto h = static_cast<uint32_t>(std::max(1.0, std::round(t.height() / nres)));
			fetchTile(t, w, h, nres, false, 100000 + k);
			uint32_t d = ++done;
			progress.set(float(d) / float(ntotal), std::format("{} : tuile native {}/{}", info_.name, d, ntotal));
		});
	}

	if (progress.cancelled() || !any.load())
		return {};
	if (info_.verticalOffset != 0.0f)
		out.shift(info_.verticalOffset);
	return out;
}

// ---------------------------------------------------------------------------------------------
// Copernicus DEM GLO-30
// ---------------------------------------------------------------------------------------------

CopernicusSource::CopernicusSource()
{
	info_.id             = "cop30";
	info_.name           = "Copernicus DEM GLO-30 (30 m, monde)";
	info_.provider       = "ESA / Copernicus";
	info_.licence        = "Copernicus DEM licence (gratuite, mention obligatoire)";
	info_.attribution    = "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
	                       "by the European Union and ESA; all rights reserved";
	info_.kind           = SourceKind::Terrain;
	info_.resolution     = 30.0;
	info_.verticalOffset = 2.3f;        // EGM2008 -> DNG, approximate
	info_.online         = true;
	info_.extentLonLat   = {-180.0, -90.0, 180.0, 90.0};
}

Grid CopernicusSource::fetch(BBox const &box, double cell, int epsg, Progress &progress, std::string *error)
{
	Grid out(box, cell, epsg);
	BBox ll = transformBox(box, epsg, EPSG_WGS84);
	int  lon0 = static_cast<int>(std::floor(ll.xmin)), lon1 = static_cast<int>(std::floor(ll.xmax));
	int  lat0 = static_cast<int>(std::floor(ll.ymin)), lat1 = static_cast<int>(std::floor(ll.ymax));
	uint32_t nlon = static_cast<uint32_t>(lon1 - lon0 + 1), total = nlon * static_cast<uint32_t>(lat1 - lat0 + 1);
	bool     any  = false;
	// Sequential on purpose: a GDAL /vsicurl read performed in a short-lived thread deadlocks that thread's exit
	// on Windows (DLL thread-detach, observed with GDAL 3.12 + curl/schannel); the few 1-degree tiles are read
	// by windowed HTTP range requests and take a second or two each anyway.
	for (uint32_t k = 0; k < total && !progress.cancelled(); ++k)
	{
		int         lon  = lon0 + int(k % nlon), lat = lat0 + int(k / nlon);
		std::string name = std::format("Copernicus_DSM_COG_10_{}{:02d}_00_{}{:03d}_00_DEM", lat >= 0 ? "N" : "S", std::abs(lat),
		                               lon >= 0 ? "E" : "W", std::abs(lon));
		std::string path = std::format("/vsicurl/https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{}/{}.tif", name, name);
		progress.set(float(k) / float(total), std::format("Copernicus : dalle {}", name.substr(23)));
		std::string err;
		Grid        part = readWindow(path, box, cell, epsg, &err);
		if (part.v.empty())
		{
			log::info("Copernicus {}: {}", name, err);        // missing tiles over the sea are normal
			continue;
		}
		out.fillFrom(part);
		any = true;
	}
	if (progress.cancelled())
		return {};
	if (!any)
	{
		if (error)
			*error = "aucune dalle Copernicus disponible";
		return {};
	}
	out.shift(info_.verticalOffset);
	return out;
}

// ---------------------------------------------------------------------------------------------

std::vector<std::unique_ptr<RasterSource>> makeOnlineSources()
{
	std::vector<std::unique_ptr<RasterSource>> v;

	// Vertical offsets to DNG (TAW), approximate (EVRF2007 national offsets): NAP +2.32, DHHN2016 +2.31,
	// NGF-IGN69 +1.82, TAW 0.
	{
		SourceInfo i{.id = "dhmv2", .name = "Flandre : DHMV II MNT 1 m (LiDAR)", .provider = "Digitaal Vlaanderen",
		             .licence = "Modellicentie gratis hergebruik", .attribution = "Bron: Digitaal Vlaanderen",
		             .resolution = 1.0, .verticalOffset = 0.0f, .extentLonLat = {2.5, 50.67, 5.95, 51.52}};
		OgcService s{.protocol = OgcService::Protocol::Wcs100, .baseUrl = "https://geo.api.vlaanderen.be/DHMV/wcs",
		             .layer = "DHMVII_DTM_1m", .requestEpsg = 31370, .outputL72 = true, .maxPixels = 2500, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "ahn4", .name = "Pays-Bas : AHN4 MNT 0,5 m (LiDAR)", .provider = "PDOK / Rijkswaterstaat",
		             .licence = "CC0", .attribution = "AHN4, PDOK (CC0)", .resolution = 0.5, .verticalOffset = 2.32f,
		             .extentLonLat = {3.3, 50.75, 7.25, 53.6}};
		OgcService s{.protocol = OgcService::Protocol::Wcs201,
		             .baseUrl  = "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wcs/v1_0",
		             .layer = "dtm_05m", .requestEpsg = 28992, .outputL72 = true, .maxPixels = 4000, .nativeRes = 0.5};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "nrw_dgm1", .name = "Allemagne (NRW) : DGM1 1 m (LiDAR)", .provider = "Geobasis NRW",
		             .licence = "dl-de/zero-2.0", .attribution = "Geobasis NRW (dl-de/zero-2.0)", .resolution = 1.0,
		             .verticalOffset = 2.31f, .extentLonLat = {5.86, 50.32, 9.47, 52.53}};
		OgcService s{.protocol = OgcService::Protocol::Wcs201, .baseUrl = "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm",
		             .layer = "nw_dgm", .requestEpsg = 25832, .outputL72 = true, .maxPixels = 2000, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "ign_rge", .name = "France : RGE ALTI 1 m", .provider = "IGN (Géoplateforme)",
		             .licence = "Licence Ouverte Etalab 2.0", .attribution = "IGN - RGE ALTI, Licence Ouverte Etalab 2.0",
		             .resolution = 1.0, .verticalOffset = 1.82f, .extentLonLat = {-5.2, 41.3, 9.6, 51.1}};
		OgcService s{.protocol = OgcService::Protocol::WmsGeoTiff, .baseUrl = "https://data.geopf.fr/wms-r/wms",
		             .layer = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", .requestEpsg = 2154, .outputL72 = false,
		             .maxPixels = 4000, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	v.push_back(std::make_unique<CopernicusSource>());
	return v;
}

}        // namespace swe::geo
