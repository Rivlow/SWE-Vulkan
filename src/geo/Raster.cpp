#include "geo/Raster.h"

#include "util/Log.h"

#include <cpl_conv.h>
#include <cpl_string.h>
#include <gdal_priv.h>
#include <gdal_utils.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>

#include <algorithm>
#include <cmath>
#include <memory>

namespace swe::geo {

namespace {

struct DatasetCloser
{
	void operator()(GDALDataset *ds) const
	{
		if (ds)
			GDALClose(ds);
	}
};
using DatasetPtr = std::unique_ptr<GDALDataset, DatasetCloser>;

DatasetPtr openRaster(std::string const &path)
{
	return DatasetPtr(GDALDataset::FromHandle(GDALOpenEx(path.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr)));
}

std::string lastError()
{
	char const *msg = CPLGetLastErrorMsg();
	return msg && *msg ? msg : "unknown GDAL error";
}

int epsgOf(GDALDataset &ds)
{
	auto const *srs = ds.GetSpatialRef();
	if (!srs)
		return 0;
	char const *auth = srs->GetAuthorityName(nullptr);
	char const *code = srs->GetAuthorityCode(nullptr);
	if (auth && code && std::string(auth) == "EPSG")
		return std::atoi(code);
	// try to identify (e.g. WKT without authority, like the SPW bathymetry)
	OGRSpatialReference copy(*srs);
	if (copy.AutoIdentifyEPSG() == OGRERR_NONE)
	{
		code = copy.GetAuthorityCode(nullptr);
		if (code)
			return std::atoi(code);
	}
	// Belgian Lambert 72 written without authority (SPW products): recognise it by its parameters or datum name
	OGRSpatialReference l72;
	l72.importFromEPSG(EPSG_LAMBERT72);
	if (copy.IsSame(&l72))
		return EPSG_LAMBERT72;
	char const *datum = srs->GetAttrValue("DATUM");
	char const *proj  = srs->GetAttrValue("PROJECTION");
	std::string d     = datum ? datum : "", p = proj ? proj : "";
	if (p.find("Lambert_Conformal_Conic") != std::string::npos &&
	    (d.find("Belge") != std::string::npos || d.find("BD72") != std::string::npos || d.find("Belgian") != std::string::npos))
		return EPSG_LAMBERT72;
	return 0;
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------------------------

Grid::Grid(BBox const &box, double cellSize, float nodataValue) :
    x0(box.xmin), y0(box.ymin), cell(cellSize), nodata(nodataValue)
{
	nx = static_cast<uint32_t>(std::max(1.0, std::round(box.width() / cellSize)));
	ny = static_cast<uint32_t>(std::max(1.0, std::round(box.height() / cellSize)));
	v.assign(cells(), nodataValue);
}

bool Grid::isNoData(float f) const
{
	return std::isnan(f) || (!std::isnan(nodata) && f == nodata) || std::abs(f) > 1e30f;
}

size_t Grid::countNoData() const
{
	return static_cast<size_t>(std::count_if(v.begin(), v.end(), [this](float f) { return isNoData(f); }));
}

double Grid::coverage() const
{
	return v.empty() ? 0.0 : 1.0 - double(countNoData()) / double(v.size());
}

void Grid::fillFrom(Grid const &src)
{
	if (src.v.size() != v.size())
		return;
	for (size_t k = 0; k < v.size(); ++k)
		if (isNoData(v[k]) && !src.isNoData(src.v[k]))
			v[k] = src.v[k];
}

void Grid::overrideFrom(Grid const &src)
{
	if (src.v.size() != v.size())
		return;
	for (size_t k = 0; k < v.size(); ++k)
		if (!src.isNoData(src.v[k]))
			v[k] = src.v[k];
}

void Grid::shift(float dz)
{
	for (auto &f : v)
		if (!isNoData(f))
			f += dz;
}

std::vector<uint8_t> Grid::fillHoles()
{
	std::vector<uint8_t> hole(v.size());
	std::vector<uint8_t> known(v.size());
	for (size_t k = 0; k < v.size(); ++k)
	{
		hole[k]  = isNoData(v[k]) ? 1 : 0;
		known[k] = !hole[k];
	}
	if (std::none_of(known.begin(), known.end(), [](uint8_t k) { return k; }))
		return hole;
	bool remaining = true;
	while (remaining)
	{
		auto prevKnown = known;
		auto prev      = v;
		remaining      = false;
		for (uint32_t i = 0; i < nx; ++i)
			for (uint32_t j = 0; j < ny; ++j)
			{
				size_t k = size_t(i) * ny + j;
				if (prevKnown[k])
					continue;
				float best = -std::numeric_limits<float>::infinity();
				auto  look = [&](int ii, int jj) {
					if (ii < 0 || jj < 0 || ii >= int(nx) || jj >= int(ny))
						return;
					size_t kk = size_t(ii) * ny + jj;
					if (prevKnown[kk])
						best = std::max(best, prev[kk]);
				};
				look(int(i) - 1, j);
				look(int(i) + 1, j);
				look(i, int(j) - 1);
				look(i, int(j) + 1);
				if (std::isfinite(best))
				{
					v[k]     = best;
					known[k] = 1;
				}
				else
					remaining = true;
			}
	}
	return hole;
}

Grid Grid::fromGdalRows(std::vector<float> const &rows, uint32_t width, uint32_t height, BBox const &box, double cellSize,
                        float nodataValue)
{
	Grid g;
	g.nx     = width;
	g.ny     = height;
	g.x0     = box.xmin;
	g.y0     = box.ymin;
	g.cell   = cellSize;
	g.nodata = nodataValue;
	g.v.resize(size_t(width) * height);
	for (uint32_t row = 0; row < height; ++row)
	{
		uint32_t j = height - 1 - row;        // first GDAL row = north edge
		for (uint32_t i = 0; i < width; ++i)
			g.v[size_t(i) * height + j] = rows[size_t(row) * width + i];
	}
	return g;
}

std::vector<float> Grid::toGdalRows() const
{
	std::vector<float> rows(v.size());
	for (uint32_t row = 0; row < ny; ++row)
	{
		uint32_t j = ny - 1 - row;
		for (uint32_t i = 0; i < nx; ++i)
			rows[size_t(row) * nx + i] = v[size_t(i) * ny + j];
	}
	return rows;
}

// ---------------------------------------------------------------------------------------------
// GDAL access
// ---------------------------------------------------------------------------------------------

bool readInfo(std::string const &path, RasterInfo &info, std::string *error)
{
	init();
	auto ds = openRaster(path);
	if (!ds)
	{
		if (error)
			*error = lastError();
		return false;
	}
	double gt[6];
	if (ds->GetGeoTransform(gt) != CE_None)
	{
		if (error)
			*error = "no georeferencing";
		return false;
	}
	info.path   = path;
	info.width  = ds->GetRasterXSize();
	info.height = ds->GetRasterYSize();
	info.bands  = ds->GetRasterCount();
	info.resX   = std::abs(gt[1]);
	info.resY   = std::abs(gt[5]);
	double x1 = gt[0] + gt[1] * info.width, y1 = gt[3] + gt[5] * info.height;
	info.bounds = {std::min(gt[0], x1), std::min(gt[3], y1), std::max(gt[0], x1), std::max(gt[3], y1)};
	info.epsg   = epsgOf(*ds);
	if (info.bands > 0)
	{
		auto *band      = ds->GetRasterBand(1);
		int   has       = 0;
		double nd       = band->GetNoDataValue(&has);
		info.hasNoData  = has != 0;
		info.nodata     = has ? static_cast<float>(nd) : std::numeric_limits<float>::quiet_NaN();
		info.dataType   = GDALGetDataTypeName(band->GetRasterDataType());
	}
	char const *desc = ds->GetDescription();
	info.description = desc ? desc : "";
	if (info.epsg == EPSG_LAMBERT72)
		info.boundsL72 = info.bounds;
	else if (info.epsg != 0)
	{
		info.boundsL72 = BBox{1e300, 1e300, -1e300, -1e300};
		double xs[3] = {info.bounds.xmin, 0.5 * (info.bounds.xmin + info.bounds.xmax), info.bounds.xmax};
		double ys[3] = {info.bounds.ymin, 0.5 * (info.bounds.ymin + info.bounds.ymax), info.bounds.ymax};
		for (double x : xs)
			for (double y : ys)
			{
				double px = x, py = y;
				if (transform(info.epsg, EPSG_LAMBERT72, px, py))
				{
					info.boundsL72.xmin = std::min(info.boundsL72.xmin, px);
					info.boundsL72.xmax = std::max(info.boundsL72.xmax, px);
					info.boundsL72.ymin = std::min(info.boundsL72.ymin, py);
					info.boundsL72.ymax = std::max(info.boundsL72.ymax, py);
				}
			}
	}
	return true;
}

Grid readWindow(std::string const &path, BBox const &box, double cell, std::string *error)
{
	init();
	auto src = openRaster(path);
	if (!src)
	{
		if (error)
			*error = lastError();
		return {};
	}
	RasterInfo info;
	readInfo(path, info);

	// output grid aligned on `cell`, snapped so cell edges fall on multiples of cell
	Grid     target(box, cell);
	uint32_t w = target.nx, h = target.ny;
	BBox     out = target.bounds();

	// Coarsening a fine raster: average; refining: bilinear (smooth relief, no new information)
	bool        coarsen = cell > 1.5 * std::max(info.resX, 1e-9);
	char const *resampling = coarsen ? "average" : "bilinear";

	// GDALWarp to a MEM dataset in Lambert 72 on exactly the target grid
	std::vector<std::string> args{"-of",  "MEM", "-t_srs", "EPSG:31370", "-te", std::to_string(out.xmin), std::to_string(out.ymin),
	                              std::to_string(out.xmax), std::to_string(out.ymax), "-ts", std::to_string(w), std::to_string(h),
	                              "-r", resampling, "-ot", "Float32", "-dstnodata", "-9999", "-wo", "INIT_DEST=NO_DATA"};
	if (info.epsg == 0)
	{
		// no usable CRS: assume Lambert 72 (SPW products sometimes lack the authority code)
		args.insert(args.end(), {"-s_srs", "EPSG:31370"});
	}
	if (!info.hasNoData && info.dataType == "Float32")
		args.insert(args.end(), {"-srcnodata", "nan"});
	std::vector<char *> argv;
	for (auto &a : args)
		argv.push_back(a.data());
	argv.push_back(nullptr);

	auto *options = GDALWarpAppOptionsNew(argv.data(), nullptr);
	if (!options)
	{
		if (error)
			*error = "invalid warp options";
		return {};
	}
	int          err   = 0;
	GDALDatasetH hSrc  = GDALDataset::ToHandle(src.get());
	auto        *dstDs = GDALDataset::FromHandle(GDALWarp("", nullptr, 1, &hSrc, options, &err));
	GDALWarpAppOptionsFree(options);
	DatasetPtr dst(dstDs);
	if (!dst || err)
	{
		if (error)
			*error = lastError();
		return {};
	}

	std::vector<float> rows(size_t(w) * h);
	if (dst->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, int(w), int(h), rows.data(), int(w), int(h), GDT_Float32, 0, 0) != CE_None)
	{
		if (error)
			*error = lastError();
		return {};
	}
	Grid g = Grid::fromGdalRows(rows, w, h, out, cell, -9999.0f);
	// normalise NoData to NaN so later merges do not care about the source convention
	for (auto &f : g.v)
		if (g.isNoData(f))
			f = std::numeric_limits<float>::quiet_NaN();
	g.nodata = std::numeric_limits<float>::quiet_NaN();
	return g;
}

bool writeGeoTiff(std::string const &path, Grid const &grid, std::string *error)
{
	init();
	auto *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (!driver)
	{
		if (error)
			*error = "GTiff driver missing";
		return false;
	}
	char **opts = nullptr;
	opts        = CSLSetNameValue(opts, "COMPRESS", "DEFLATE");
	opts        = CSLSetNameValue(opts, "TILED", "YES");
	DatasetPtr ds(driver->Create(path.c_str(), int(grid.nx), int(grid.ny), 1, GDT_Float32, opts));
	CSLDestroy(opts);
	if (!ds)
	{
		if (error)
			*error = lastError();
		return false;
	}
	double gt[6] = {grid.x0, grid.cell, 0.0, grid.y0 + grid.ny * grid.cell, 0.0, -grid.cell};
	ds->SetGeoTransform(gt);
	OGRSpatialReference srs;
	srs.importFromEPSG(EPSG_LAMBERT72);
	ds->SetSpatialRef(&srs);
	auto rows = grid.toGdalRows();
	for (auto &f : rows)
		if (std::isnan(f))
			f = -9999.0f;
	auto *band = ds->GetRasterBand(1);
	band->SetNoDataValue(-9999.0);
	if (band->RasterIO(GF_Write, 0, 0, int(grid.nx), int(grid.ny), rows.data(), int(grid.nx), int(grid.ny), GDT_Float32, 0, 0) != CE_None)
	{
		if (error)
			*error = lastError();
		return false;
	}
	return true;
}

}        // namespace swe::geo
