#include "geo/Tiles.h"

#include "geo/Http.h"
#include "util/Log.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace swe::geo {

Image decodeImage(std::vector<uint8_t> const &bytes)
{
	Image img;
	int   w = 0, h = 0, n = 0;
	auto *px = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &n, 4);
	if (!px)
		return img;
	img.width  = static_cast<uint32_t>(w);
	img.height = static_cast<uint32_t>(h);
	img.rgba.assign(px, px + size_t(w) * h * 4);
	stbi_image_free(px);
	return img;
}

namespace {

// Web Mercator bbox of tile (z, x, y)
BBox tileBounds3857(int z, int x, int y)
{
	auto a = mercator(fromTile({double(x), double(y + 1)}, z));
	auto b = mercator(fromTile({double(x + 1), double(y)}, z));
	return {a.x, a.y, b.x, b.y};
}

constexpr char const *kSpwOrtho = "https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_LAST/MapServer/WMSServer";

std::string wmsUrl(std::string const &base, std::string const &layer, int epsg, BBox const &b, uint32_t w, uint32_t h,
                   char const *format)
{
	// WMS 1.3.0 with EPSG:3857 / 31370: axis order x, y
	return std::format("{}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={}&STYLES=&CRS=EPSG:{}&BBOX={:.3f},{:.3f},{:.3f},{:.3f}"
	                   "&WIDTH={}&HEIGHT={}&FORMAT={}",
	                   base, layer, epsg, b.xmin, b.ymin, b.xmax, b.ymax, w, h, format);
}

}        // namespace

TileLayer osmLayer()
{
	TileLayer l;
	l.id          = "osm";
	l.name        = "OpenStreetMap";
	l.attribution = "© OpenStreetMap contributors";
	l.maxZoom     = 19;
	l.url         = [](int z, int x, int y) { return std::format("https://tile.openstreetmap.org/{}/{}/{}.png", z, x, y); };
	return l;
}

TileLayer spwOrthoLayer()
{
	TileLayer l;
	l.id          = "spw_ortho";
	l.name        = "Orthophoto SPW (Wallonie)";
	l.attribution = "Orthophotos © SPW";
	l.minZoom     = 8;
	l.maxZoom     = 20;
	l.url         = [](int z, int x, int y) { return wmsUrl(kSpwOrtho, "0", EPSG_MERCATOR, tileBounds3857(z, x, y), 256, 256, "image/jpeg"); };
	return l;
}

// ---------------------------------------------------------------------------------------------

TileProvider::TileProvider(std::vector<TileLayer> layers, unsigned workers) :
    layers_(std::move(layers))
{
	for (unsigned k = 0; k < std::max(1u, workers); ++k)
		threads_.emplace_back([this] { worker(); });
}

TileProvider::~TileProvider()
{
	{
		std::lock_guard lock(mutex_);
		stop_ = true;
	}
	cv_.notify_all();
}

TileState TileProvider::get(TileKey const &key, Image const **image)
{
	std::lock_guard lock(mutex_);
	auto            it = ready_.find(key);
	if (it != ready_.end())
	{
		if (image)
			*image = it->second.get();
		return TileState::Ready;
	}
	auto st = state_.find(key);
	if (st != state_.end())
		return st->second;
	// not seen: queue it (front: the most recently requested tiles are the visible ones)
	state_[key] = TileState::Pending;
	queue_.push_front(key);
	cv_.notify_one();
	return TileState::Pending;
}

void TileProvider::cancelPending()
{
	std::lock_guard lock(mutex_);
	for (auto const &k : queue_)
		state_.erase(k);
	queue_.clear();
}

size_t TileProvider::pendingCount() const
{
	std::lock_guard lock(mutex_);
	return queue_.size();
}

void TileProvider::worker()
{
	for (;;)
	{
		TileKey key;
		{
			std::unique_lock lock(mutex_);
			cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
			if (stop_)
				return;
			key = queue_.front();
			queue_.pop_front();
		}
		auto const &layer = layers_[static_cast<size_t>(key.layer)];
		auto        url   = layer.url(key.z, key.x, key.y);
		auto        resp  = http::get(url, true, 30);
		Image       img;
		if (resp.ok())
			img = decodeImage(resp.body);
		std::lock_guard lock(mutex_);
		if (img.empty())
		{
			state_[key] = TileState::Failed;
			continue;
		}
		ready_[key] = std::make_unique<Image>(std::move(img));
		state_[key] = TileState::Ready;
		order_.push_back(key);
		while (order_.size() > maxReady_)
		{
			auto old = order_.front();
			order_.pop_front();
			ready_.erase(old);
			state_.erase(old);
		}
	}
}

// ---------------------------------------------------------------------------------------------

Image fetchOrthophoto(BBox const &box, uint32_t pixels, std::string *error)
{
	double   longest = std::max(box.width(), box.height());
	double   res     = longest / pixels;                 // m / px
	uint32_t W       = static_cast<uint32_t>(std::round(box.width() / res));
	uint32_t H       = static_cast<uint32_t>(std::round(box.height() / res));
	Image    out;
	out.width  = W;
	out.height = H;
	out.rgba.assign(size_t(W) * H * 4, 0);

	constexpr uint32_t kTile = 2048;
	uint32_t           nx = (W + kTile - 1) / kTile, ny = (H + kTile - 1) / kTile;
	bool               any = false;
	for (uint32_t ty = 0; ty < ny; ++ty)
		for (uint32_t tx = 0; tx < nx; ++tx)
		{
			uint32_t px0 = tx * kTile, py0 = ty * kTile;        // py from the top (north)
			uint32_t w = std::min(kTile, W - px0), h = std::min(kTile, H - py0);
			BBox     b{box.xmin + px0 * res, box.ymax - (py0 + h) * res, box.xmin + (px0 + w) * res, box.ymax - py0 * res};
			auto     resp = http::get(wmsUrl(kSpwOrtho, "0", EPSG_LAMBERT72, b, w, h, "image/jpeg"), true, 120);
			if (!resp.ok())
			{
				if (error)
					*error = resp.error.empty() ? std::format("HTTP {}", resp.status) : resp.error;
				continue;
			}
			Image part = decodeImage(resp.body);
			if (part.empty() || part.width != w || part.height != h)
			{
				if (error)
					*error = "réponse WMS non décodable";
				continue;
			}
			for (uint32_t row = 0; row < h; ++row)
				std::copy_n(&part.rgba[size_t(row) * w * 4], size_t(w) * 4, &out.rgba[(size_t(py0 + row) * W + px0) * 4]);
			any = true;
		}
	if (!any)
		return {};
	return out;
}

}        // namespace swe::geo
