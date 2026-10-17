#include "geo/LocalLibrary.h"

#include "util/Log.h"

#include <cpl_string.h>
#include <cpl_vsi.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace swe::geo {

namespace fs = std::filesystem;

namespace {

std::string upper(std::string s)
{
	for (auto &c : s)
		c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
	return s;
}

SourceKind guessKind(std::string const &path)
{
	auto u = upper(path);
	return u.find("BATHY") != std::string::npos ? SourceKind::Bathymetry : SourceKind::Terrain;
}

}        // namespace

LocalLibrary::LocalLibrary(fs::path indexFile) :
    indexFile_(std::move(indexFile))
{
	load();
}

void LocalLibrary::addFolder(fs::path const &dir)
{
	if (std::ranges::find(folders_, dir) == folders_.end())
		folders_.push_back(dir);
}

void LocalLibrary::removeFolder(size_t index)
{
	if (index < folders_.size())
		folders_.erase(folders_.begin() + static_cast<std::ptrdiff_t>(index));
}

void LocalLibrary::indexFile(fs::path const &file)
{
	std::vector<std::string> paths;
	auto                     ext = upper(file.extension().string());
	if (ext == ".TIF" || ext == ".TIFF")
		paths.push_back(file.string());
	else if (ext == ".ZIP")
	{
		// GDAL lists the zip content; keep the GeoTIFFs
		std::string vsi = "/vsizip/" + file.string();
		char      **list = VSIReadDirRecursive(vsi.c_str());
		for (int k = 0; list && list[k]; ++k)
		{
			auto e = upper(fs::path(list[k]).extension().string());
			if (e == ".TIF" || e == ".TIFF")
				paths.push_back(vsi + "/" + list[k]);
		}
		CSLDestroy(list);
	}
	for (auto const &p : paths)
	{
		if (std::ranges::any_of(entries_, [&](LibraryEntry const &e) { return e.info.path == p; }))
			continue;
		LibraryEntry e;
		std::string  err;
		if (!readInfo(p, e.info, &err))
		{
			log::warn("library: {} ignored ({})", p, err);
			continue;
		}
		if (e.info.bands < 1 || e.info.epsg == 0 && e.info.boundsL72.empty())
			continue;
		e.kind  = guessKind(p);
		e.label = fs::path(p).filename().string();
		entries_.push_back(std::move(e));
	}
}

void LocalLibrary::rescan(Progress *progress)
{
	entries_.clear();
	std::vector<fs::path> files;
	for (auto const &dir : folders_)
	{
		std::error_code ec;
		if (!fs::exists(dir, ec))
			continue;
		if (fs::is_regular_file(dir, ec))
		{
			files.push_back(dir);
			continue;
		}
		for (auto const &entry : fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec))
			if (entry.is_regular_file(ec))
			{
				auto ext = upper(entry.path().extension().string());
				if (ext == ".TIF" || ext == ".TIFF" || ext == ".ZIP")
					files.push_back(entry.path());
			}
	}
	for (size_t k = 0; k < files.size(); ++k)
	{
		if (progress)
			progress->set(float(k) / float(files.size()), "Index : " + files[k].filename().string());
		indexFile(files[k]);
	}
	save();
	log::info("library: {} raster(s) indexed from {} folder(s)", entries_.size(), folders_.size());
}

void LocalLibrary::load()
{
	std::ifstream f(indexFile_);
	if (!f)
		return;
	try
	{
		auto j = nlohmann::json::parse(f);
		for (auto const &d : j.value("folders", nlohmann::json::array()))
			folders_.push_back(fs::path(d.get<std::string>()));
		for (auto const &e : j.value("entries", nlohmann::json::array()))
		{
			LibraryEntry le;
			le.info.path       = e.at("path").get<std::string>();
			le.info.epsg       = e.value("epsg", 0);
			auto b             = e.at("bounds_l72");
			le.info.boundsL72  = {b[0], b[1], b[2], b[3]};
			auto bb            = e.value("bounds", b);
			le.info.bounds     = {bb[0], bb[1], bb[2], bb[3]};
			le.info.resX       = e.value("res_x", 1.0);
			le.info.resY       = e.value("res_y", 1.0);
			le.info.width      = e.value("width", 0u);
			le.info.height     = e.value("height", 0u);
			le.info.hasNoData  = e.value("has_nodata", false);
			le.info.nodata     = e.value("nodata", std::numeric_limits<float>::quiet_NaN());
			le.info.bands      = e.value("bands", 1);
			le.info.dataType   = e.value("dtype", "Float32");
			le.kind            = e.value("kind", "terrain") == "bathymetry" ? SourceKind::Bathymetry : SourceKind::Terrain;
			le.verticalOffset  = e.value("vertical_offset", 0.0f);
			le.label           = e.value("label", fs::path(le.info.path).filename().string());
			le.enabled         = e.value("enabled", true);
			entries_.push_back(std::move(le));
		}
	}
	catch (std::exception const &ex)
	{
		log::warn("{}: {}", indexFile_.string(), ex.what());
	}
}

void LocalLibrary::save() const
{
	nlohmann::json j;
	for (auto const &d : folders_)
		j["folders"].push_back(d.string());
	j["entries"] = nlohmann::json::array();
	for (auto const &e : entries_)
	{
		auto const &i = e.info;
		j["entries"].push_back({{"path", i.path},
		                        {"epsg", i.epsg},
		                        {"bounds", {i.bounds.xmin, i.bounds.ymin, i.bounds.xmax, i.bounds.ymax}},
		                        {"bounds_l72", {i.boundsL72.xmin, i.boundsL72.ymin, i.boundsL72.xmax, i.boundsL72.ymax}},
		                        {"res_x", i.resX},
		                        {"res_y", i.resY},
		                        {"width", i.width},
		                        {"height", i.height},
		                        {"has_nodata", i.hasNoData},
		                        {"nodata", std::isnan(i.nodata) ? 0.0f : i.nodata},
		                        {"bands", i.bands},
		                        {"dtype", i.dataType},
		                        {"kind", e.kind == SourceKind::Bathymetry ? "bathymetry" : "terrain"},
		                        {"vertical_offset", e.verticalOffset},
		                        {"label", e.label},
		                        {"enabled", e.enabled}});
	}
	std::error_code ec;
	fs::create_directories(indexFile_.parent_path(), ec);
	std::ofstream(indexFile_) << j.dump(2) << '\n';
}

// ---------------------------------------------------------------------------------------------

LocalSource::LocalSource(LocalLibrary const &library, SourceKind kind) :
    library_(library), kind_(kind)
{
	info_.id          = kind == SourceKind::Bathymetry ? "local_bathy" : "local_dtm";
	info_.name        = kind == SourceKind::Bathymetry ? "Bathymétrie locale (GeoTIFF)" : "MNT local (GeoTIFF)";
	info_.provider    = "fichiers locaux";
	info_.licence     = "selon la source (SPW : CC BY 4.0)";
	info_.attribution = "Source : Service public de Wallonie (SPW)";
	info_.kind        = kind;
	info_.online      = false;
	refresh();
}

void LocalSource::refresh()
{
	BBox   e{1e300, 1e300, -1e300, -1e300};
	double res = 0.0;
	bool   any = false;
	for (auto const &le : library_.entries())
	{
		if (le.kind != kind_ || !le.enabled)
			continue;
		auto const &b = le.info.boundsL72;
		e.xmin        = std::min(e.xmin, b.xmin);
		e.ymin        = std::min(e.ymin, b.ymin);
		e.xmax        = std::max(e.xmax, b.xmax);
		e.ymax        = std::max(e.ymax, b.ymax);
		res           = any ? std::min(res, le.info.resX) : le.info.resX;
		any           = true;
	}
	info_.extentL72  = any ? e : BBox{};
	info_.resolution = any ? res : 1.0;
}

Grid LocalSource::fetch(BBox const &box, double cell, Progress &progress, std::string *error)
{
	Grid out(box, cell);
	// finest rasters first: they win where several overlap
	std::vector<LibraryEntry const *> hits;
	for (auto const &le : library_.entries())
		if (le.kind == kind_ && le.enabled && !le.info.boundsL72.intersect(box).empty())
			hits.push_back(&le);
	std::ranges::sort(hits, [](auto *a, auto *b) { return a->info.resX < b->info.resX; });
	if (hits.empty())
	{
		if (error)
			*error = "aucun raster local ne couvre la zone";
		return {};
	}
	for (size_t k = 0; k < hits.size(); ++k)
	{
		if (progress.cancelled())
			return {};
		progress.set(float(k) / float(hits.size()), info_.name + " : " + hits[k]->label);
		std::string err;
		Grid        part = readWindow(hits[k]->info.path, box, cell, &err);
		if (part.v.empty())
		{
			log::warn("{}: {}", hits[k]->label, err);
			continue;
		}
		if (hits[k]->verticalOffset != 0.0f)
			part.shift(hits[k]->verticalOffset);
		out.fillFrom(part);
	}
	return out;
}

}        // namespace swe::geo
