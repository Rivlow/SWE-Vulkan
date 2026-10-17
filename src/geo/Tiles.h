#pragma once
// Slippy-map tiles (Web Mercator, 256 px) for the area picker, and WMS images for the orthophoto drape.
// Vulkan-free: delivers RGBA pixels; the GUI uploads them. Downloads run on worker threads, responses are
// cached on disk by Http. OSM tile policy: identifying User-Agent, no bulk download, attribution displayed.
#include "geo/Crs.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace swe::geo {

struct Image
{
	uint32_t             width = 0, height = 0;
	std::vector<uint8_t> rgba;        // width * height * 4
	bool                 empty() const { return rgba.empty(); }
};

// Decodes PNG / JPEG bytes (stb_image)
Image decodeImage(std::vector<uint8_t> const &bytes);

struct TileLayer
{
	std::string id;
	std::string name;
	std::string attribution;
	int         minZoom = 0, maxZoom = 19;
	// URL of tile (z, x, y) - XYZ template or WMS GetMap in EPSG:3857 for the tile's bbox
	std::function<std::string(int z, int x, int y)> url;
};

TileLayer osmLayer();
TileLayer spwOrthoLayer();        // Wallonia orthophoto (WMS, cut into 3857 tiles)
TileLayer sentinelLayer();        // Sentinel-2 cloudless (EOX), world

struct TileKey
{
	int  z = 0, x = 0, y = 0;
	int  layer = 0;
	bool operator==(TileKey const &o) const { return z == o.z && x == o.x && y == o.y && layer == o.layer; }
};
struct TileKeyHash
{
	size_t operator()(TileKey const &k) const
	{
		return std::hash<int64_t>()((int64_t(k.layer) << 40) ^ (int64_t(k.z) << 32) ^ (int64_t(k.x) << 16) ^ k.y);
	}
};

enum class TileState
{
	Missing,
	Pending,
	Ready,
	Failed,
};

// Asynchronous tile provider. The GUI thread calls request()/get() each frame; ready tiles stay in memory
// (bounded LRU-ish cache).
class TileProvider
{
  public:
	explicit TileProvider(std::vector<TileLayer> layers, unsigned workers = 4);
	~TileProvider();

	std::vector<TileLayer> const &layers() const { return layers_; }

	// Returns the tile if ready; queues the download otherwise (at most `maxPending` in flight).
	TileState get(TileKey const &key, Image const **image);
	void      cancelPending();        // drop queued (not started) downloads - after a zoom change
	size_t    pendingCount() const;

  private:
	void worker();

	std::vector<TileLayer>                                   layers_;
	mutable std::mutex                                       mutex_;
	std::condition_variable                                  cv_;
	std::deque<TileKey>                                      queue_;
	std::unordered_map<TileKey, std::unique_ptr<Image>, TileKeyHash> ready_;
	std::unordered_map<TileKey, TileState, TileKeyHash>      state_;
	std::deque<TileKey>                                      order_;        // insertion order for eviction
	std::vector<std::jthread>                                threads_;
	bool                                                     stop_ = false;
	size_t                                                   maxReady_ = 1500;
};

// Orthophoto of `box` (in `epsg`) from the best regional WMS (SPW, Flanders, PDOK, NRW, IGN) or, anywhere in
// the world, Sentinel-2 cloudless (EOX, 10 m). Assembled from <= 2048 px requests in the server's CRS, then
// reprojected onto the box. `pixels` is the longest side. Empty image on failure; `attribution` receives the
// text to credit.
Image fetchOrthophoto(BBox const &box, int epsg, uint32_t pixels, std::string *error = nullptr, std::string *attribution = nullptr);

// Name of the orthophoto source that would be used at a location
std::string orthophotoSourceName(LonLat where);

}        // namespace swe::geo
