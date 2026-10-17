#pragma once
// Local raster library: folders of GeoTIFF (and zips of GeoTIFF) indexed once, read by window afterwards.
// This is how the Walloon products (SPW DTM / bathymetry downloads, no WCS exists) enter the application.
#include "geo/RasterSource.h"

#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace swe::geo {

struct LibraryEntry
{
	RasterInfo  info;
	SourceKind  kind = SourceKind::Terrain;
	float       verticalOffset = 0.0f;      // to DNG
	std::string label;
	bool        enabled = true;
};

class LocalLibrary
{
  public:
	explicit LocalLibrary(std::filesystem::path indexFile = "cache/library.json");

	// Folders scanned for .tif/.tiff/.zip (recursive). Scanning reads only headers.
	std::vector<std::filesystem::path> const &folders() const { return folders_; }
	void                                      addFolder(std::filesystem::path const &dir);
	void                                      removeFolder(size_t index);
	void                                      rescan(Progress *progress = nullptr);

	std::vector<LibraryEntry> const &entries() const { return entries_; }
	std::vector<LibraryEntry>       &entries() { return entries_; }

	void load();
	void save() const;

  private:
	void indexFile(std::filesystem::path const &file);

	std::filesystem::path             indexFile_;
	std::vector<std::filesystem::path> folders_;
	std::vector<LibraryEntry>         entries_;
};

// One kind of the library (terrain or bathymetry) seen as a RasterSource.
class LocalSource : public RasterSource
{
  public:
	LocalSource(LocalLibrary const &library, SourceKind kind);
	SourceInfo const &info() const override { return info_; }
	Grid              fetch(BBox const &box, double cell, Progress &progress, std::string *error) override;
	// recompute footprint / resolution from the library entries (after a rescan)
	void refresh();

  private:
	LocalLibrary const &library_;
	SourceKind          kind_;
	SourceInfo          info_;
};

}        // namespace swe::geo
