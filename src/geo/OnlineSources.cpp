#include "geo/OnlineSources.h"

#include "geo/Http.h"
#include "util/Log.h"

#include <cpl_conv.h>
#include <cpl_vsi.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace swe::geo {

namespace {

// Writes bytes into GDAL's in-memory file system and reads them as a raster window.
Grid gridFromBytes(std::vector<uint8_t> const &bytes, std::string const &name, BBox const &box, double cell, std::string *error)
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
	Grid g = readWindow(vsi, box, cell, error);
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

std::string OgcSource::requestUrl(BBox const &b, uint32_t w, uint32_t h) const
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
		if (s.outputL72)
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

Grid OgcSource::fetch(BBox const &box, double cell, Progress &progress, std::string *error)
{
	Grid out(box, cell);
	// request resolution: never finer than the native product, never coarser than the target
	double reqRes = std::max(service_.nativeRes, std::min(cell, service_.nativeRes * 4.0));
	// request box in the server's CRS (hull of the target box)
	BBox reqBox = box;
	if (service_.requestEpsg != EPSG_LAMBERT72)
	{
		reqBox = BBox{1e300, 1e300, -1e300, -1e300};
		double xs[3] = {box.xmin, 0.5 * (box.xmin + box.xmax), box.xmax};
		double ys[3] = {box.ymin, 0.5 * (box.ymin + box.ymax), box.ymax};
		for (double x : xs)
			for (double y : ys)
			{
				double px = x, py = y;
				if (!transform(EPSG_LAMBERT72, service_.requestEpsg, px, py))
				{
					if (error)
						*error = "reprojection vers le CRS du service impossible";
					return {};
				}
				reqBox.xmin = std::min(reqBox.xmin, px);
				reqBox.xmax = std::max(reqBox.xmax, px);
				reqBox.ymin = std::min(reqBox.ymin, py);
				reqBox.ymax = std::max(reqBox.ymax, py);
			}
		// margin for the warp
		double m = 2.0 * reqRes;
		reqBox   = {reqBox.xmin - m, reqBox.ymin - m, reqBox.xmax + m, reqBox.ymax + m};
	}
	else
		reqBox = {box.xmin - reqRes, box.ymin - reqRes, box.xmax + reqRes, box.ymax + reqRes};

	// tiling under the pixel limit; a WCS 2.0.1 GetCoverage without scaling returns the native resolution
	bool     native = service_.protocol == OgcService::Protocol::Wcs201;
	if (native)
		reqRes = service_.nativeRes;
	double   tileM = service_.maxPixels * reqRes * 0.98;
	uint32_t tx    = static_cast<uint32_t>(std::ceil(reqBox.width() / tileM));
	uint32_t ty    = static_cast<uint32_t>(std::ceil(reqBox.height() / tileM));
	uint32_t total = tx * ty, done = 0;
	bool     any   = false;
	for (uint32_t a = 0; a < tx; ++a)
		for (uint32_t b = 0; b < ty; ++b)
		{
			if (progress.cancelled())
				return {};
			BBox t{reqBox.xmin + a * tileM, reqBox.ymin + b * tileM, std::min(reqBox.xmin + (a + 1) * tileM, reqBox.xmax),
			       std::min(reqBox.ymin + (b + 1) * tileM, reqBox.ymax)};
			auto w   = static_cast<uint32_t>(std::max(1.0, std::round(t.width() / reqRes)));
			auto h   = static_cast<uint32_t>(std::max(1.0, std::round(t.height() / reqRes)));
			auto url = requestUrl(t, w, h);
			progress.set(float(done) / float(total), std::format("{} : tuile {}/{}", info_.name, done + 1, total));
			auto r = http::get(url);
			++done;
			if (!r.ok())
			{
				if (error)
					*error = r.error.empty() ? std::format("HTTP {}", r.status) : r.error;
				continue;
			}
			auto tiff = extractTiff(r.body);
			if (tiff.empty())
			{
				// probably an XML exception: keep it short in the log
				std::string head(r.body.begin(), r.body.begin() + std::min<size_t>(r.body.size(), 300));
				log::warn("{}: réponse non TIFF: {}", info_.name, head);
				if (error)
					*error = "réponse du serveur non exploitable";
				continue;
			}
			std::string err;
			Grid        part = gridFromBytes(tiff, info_.id, box, cell, &err);
			if (part.v.empty())
			{
				log::warn("{}: {}", info_.name, err);
				continue;
			}
			out.fillFrom(part);
			any = true;
		}
	if (!any)
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
	info_.extentL72      = {-1e7, -1e7, 1e7, 1e7};
}

Grid CopernicusSource::fetch(BBox const &box, double cell, Progress &progress, std::string *error)
{
	Grid out(box, cell);
	BBox ll = lambertToLonLatBox(box);
	int  lon0 = static_cast<int>(std::floor(ll.xmin)), lon1 = static_cast<int>(std::floor(ll.xmax));
	int  lat0 = static_cast<int>(std::floor(ll.ymin)), lat1 = static_cast<int>(std::floor(ll.ymax));
	int  total = (lon1 - lon0 + 1) * (lat1 - lat0 + 1), done = 0;
	bool any = false;
	for (int lat = lat0; lat <= lat1; ++lat)
		for (int lon = lon0; lon <= lon1; ++lon)
		{
			if (progress.cancelled())
				return {};
			std::string name = std::format("Copernicus_DSM_COG_10_{}{:02d}_00_{}{:03d}_00_DEM", lat >= 0 ? "N" : "S", std::abs(lat),
			                               lon >= 0 ? "E" : "W", std::abs(lon));
			std::string path = std::format("/vsicurl/https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{}/{}.tif", name, name);
			progress.set(float(done) / float(total), std::format("Copernicus : dalle {}", name.substr(23)));
			std::string err;
			Grid        part = readWindow(path, box, cell, &err);
			++done;
			if (part.v.empty())
			{
				log::info("Copernicus {}: {}", name, err);        // missing tiles over the sea are normal
				continue;
			}
			out.fillFrom(part);
			any = true;
		}
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
		             .resolution = 1.0, .verticalOffset = 0.0f, .extentL72 = {20000, 150000, 262000, 246000}};
		OgcService s{.protocol = OgcService::Protocol::Wcs100, .baseUrl = "https://geo.api.vlaanderen.be/DHMV/wcs",
		             .layer = "DHMVII_DTM_1m", .requestEpsg = 31370, .outputL72 = true, .maxPixels = 2500, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "ahn4", .name = "Pays-Bas : AHN4 MNT 0,5 m (LiDAR)", .provider = "PDOK / Rijkswaterstaat",
		             .licence = "CC0", .attribution = "AHN4, PDOK (CC0)", .resolution = 0.5, .verticalOffset = 2.32f,
		             .extentL72 = {60000, 160000, 320000, 490000}};
		OgcService s{.protocol = OgcService::Protocol::Wcs201,
		             .baseUrl  = "https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wcs/v1_0",
		             .layer = "dtm_05m", .requestEpsg = 31370, .outputL72 = true, .maxPixels = 4000, .nativeRes = 0.5};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "nrw_dgm1", .name = "Allemagne (NRW) : DGM1 1 m (LiDAR)", .provider = "Geobasis NRW",
		             .licence = "dl-de/zero-2.0", .attribution = "Geobasis NRW (dl-de/zero-2.0)", .resolution = 1.0,
		             .verticalOffset = 2.31f, .extentL72 = {266000, 85000, 430000, 310000}};
		OgcService s{.protocol = OgcService::Protocol::Wcs201, .baseUrl = "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm",
		             .layer = "nw_dgm", .requestEpsg = 31370, .outputL72 = true, .maxPixels = 2000, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	{
		SourceInfo i{.id = "ign_rge", .name = "France : RGE ALTI 1 m", .provider = "IGN (Géoplateforme)",
		             .licence = "Licence Ouverte Etalab 2.0", .attribution = "IGN - RGE ALTI, Licence Ouverte Etalab 2.0",
		             .resolution = 1.0, .verticalOffset = 1.82f, .extentL72 = {-350000, -750000, 330000, 40000}};
		OgcService s{.protocol = OgcService::Protocol::WmsGeoTiff, .baseUrl = "https://data.geopf.fr/wms-r/wms",
		             .layer = "ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES", .requestEpsg = 2154, .outputL72 = false,
		             .maxPixels = 4000, .nativeRes = 1.0};
		v.push_back(std::make_unique<OgcSource>(i, s));
	}
	v.push_back(std::make_unique<CopernicusSource>());
	return v;
}

}        // namespace swe::geo
