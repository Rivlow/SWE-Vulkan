#pragma once
// Online elevation services, all returning float32 GeoTIFF:
//   - OGC WCS 1.0.0 / 2.0.1 (Flanders DHMV, Netherlands AHN, NRW DGM1)
//   - IGN "WMS-R" GetMap in image/geotiff (France)
//   - Copernicus DEM GLO-30 cloud-optimised GeoTIFFs on public S3 (world)
// Each request is tiled under the server's pixel limit, cached on disk, warped to the target grid.
#include "geo/RasterSource.h"

namespace swe::geo {

struct OgcService
{
	enum class Protocol
	{
		Wcs100,        // GetCoverage COVERAGE/CRS/BBOX/WIDTH/HEIGHT/FORMAT=GeoTIFF
		Wcs201,        // GetCoverage COVERAGEID/SUBSET x,y (+ SUBSETTINGCRS / OUTPUTCRS)
		WmsGeoTiff,    // GetMap FORMAT=image/geotiff (IGN WMS-R)
	};
	Protocol    protocol = Protocol::Wcs201;
	std::string baseUrl;
	std::string layer;             // coverage id / layer name
	int         requestEpsg = 31370;   // CRS of the BBOX we send (the server's native grid)
	bool        outputL72 = true;      // server can reproject to Lambert 72 (OUTPUTCRS) when that is the target
	int         maxPixels = 2000;      // per side
	double      nativeRes = 1.0;
};

class OgcSource : public RasterSource
{
  public:
	OgcSource(SourceInfo info, OgcService service);
	SourceInfo const &info() const override { return info_; }
	Grid              fetch(BBox const &box, double cell, int epsg, Progress &progress, std::string *error) override;

  private:
	std::string requestUrl(BBox const &boxReq, uint32_t w, uint32_t h, bool outputL72) const;

	SourceInfo info_;
	OgcService service_;
};

class CopernicusSource : public RasterSource
{
  public:
	CopernicusSource();
	SourceInfo const &info() const override { return info_; }
	Grid              fetch(BBox const &box, double cell, int epsg, Progress &progress, std::string *error) override;

  private:
	SourceInfo info_;
};

// The configured sources, in priority order (finest / most trusted first): Flanders, NL, NRW, France, world.
std::vector<std::unique_ptr<RasterSource>> makeOnlineSources();

}        // namespace swe::geo
