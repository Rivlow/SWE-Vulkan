#include "geo/Tiles.h"

#include "geo/Http.h"
#include "util/Log.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>
#include <stb_image_write.h>

#include "geo/Raster.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <format>
#include <mutex>
#include <thread>

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
constexpr char const *kEox      = "https://tiles.maps.eox.at/wms";

// Orthophoto WMS services, finest first; the last one covers the world.
struct OrthoService
{
	char const *id;
	char const *name;
	char const *base;
	char const *layer;
	int         epsg;            // request CRS (projected, x/y axis order)
	bool        wms13;           // 1.3.0 (CRS=) or 1.1.1 (SRS=)
	char const *format;
	int         maxPixels;
	BBox        extentLonLat;
	char const *attribution;
};

OrthoService const kOrthoServices[] = {
    {"spw", "Orthophoto SPW (Wallonie, 25 cm)", kSpwOrtho, "0", 31370, true, "image/jpeg", 4096, {2.75, 49.49, 6.45, 50.86},
     "Orthophotos © Service public de Wallonie (SPW)"},
    {"omw", "Orthophoto Digitaal Vlaanderen (25 cm)", "https://geo.api.vlaanderen.be/OMWRGBMRVL/wms", "Ortho", 31370, true, "image/jpeg", 2048,
     {2.5, 50.67, 5.95, 51.52}, "Orthofoto's Digitaal Vlaanderen"},
    {"pdok", "Luchtfoto PDOK (Pays-Bas, 25 cm)", "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0", "Actueel_ortho25", 28992, true,
     "image/jpeg", 4000, {3.3, 50.75, 7.25, 53.6}, "Luchtfoto PDOK / Beeldmateriaal Nederland (CC BY 4.0)"},
    {"nrw", "DOP Geobasis NRW (10 cm)", "https://www.wms.nrw.de/geobasis/wms_nw_dop", "nw_dop_rgb", 25832, true, "image/jpeg", 4000,
     {5.86, 50.32, 9.47, 52.53}, "Digitale Orthophotos © Geobasis NRW (dl-de/zero-2.0)"},
    {"ign", "Orthophoto IGN (France, 20 cm)", "https://data.geopf.fr/wms-r/wms", "ORTHOIMAGERY.ORTHOPHOTOS", 2154, true, "image/jpeg", 4096,
     {-5.2, 41.3, 9.6, 51.1}, "Orthophotos © IGN - Géoplateforme (Licence Ouverte Etalab 2.0)"},
    {"s2", "Sentinel-2 cloudless (EOX, 10 m, monde)", kEox, "s2cloudless-2020_3857", 3857, false, "image/jpeg", 4096,
     {-180.0, -85.0, 180.0, 85.0},
     "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)"},
};

std::string wmsUrl(std::string const &base, std::string const &layer, int epsg, BBox const &b, uint32_t w, uint32_t h,
                   char const *format, bool wms13 = true)
{
	// projected CRS: x/y axis order in both versions
	std::string sep = base.find('?') == std::string::npos ? "?" : "&";
	if (wms13)
		return std::format("{}{}SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS={}&STYLES=&CRS=EPSG:{}&BBOX={:.3f},{:.3f},{:.3f},{:.3f}"
		                   "&WIDTH={}&HEIGHT={}&FORMAT={}",
		                   base, sep, layer, epsg, b.xmin, b.ymin, b.xmax, b.ymax, w, h, format);
	return std::format("{}{}SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={}&STYLES=&SRS=EPSG:{}&BBOX={:.3f},{:.3f},{:.3f},{:.3f}"
	                   "&WIDTH={}&HEIGHT={}&FORMAT={}",
	                   base, sep, layer, epsg, b.xmin, b.ymin, b.xmax, b.ymax, w, h, format);
}

OrthoService const &orthoServiceAt(LonLat where)
{
	for (auto const &s : kOrthoServices)
		if (s.extentLonLat.contains(where.lon, where.lat))
			return s;
	return kOrthoServices[std::size(kOrthoServices) - 1];
}

}        // namespace

std::string orthophotoSourceName(LonLat where)
{
	return orthoServiceAt(where).name;
}

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

TileLayer sentinelLayer()
{
	TileLayer l;
	l.id          = "s2cloudless";
	l.name        = "Satellite Sentinel-2 (EOX, monde)";
	l.attribution = "Sentinel-2 cloudless by EOX (modified Copernicus Sentinel data 2020)";
	l.minZoom     = 2;
	l.maxZoom     = 17;
	l.url = [](int z, int x, int y) { return wmsUrl(kEox, "s2cloudless-2020_3857", EPSG_MERCATOR, tileBounds3857(z, x, y), 256, 256, "image/jpeg", false); };
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

namespace {

// Mosaic of WMS requests covering `srvBox` (in the service CRS) at `res` m/px: RGBA, row 0 = north.
Image fetchMosaic(OrthoService const &s, BBox const &srvBox, double res, std::string *error)
{
	uint32_t W = static_cast<uint32_t>(std::round(srvBox.width() / res));
	uint32_t H = static_cast<uint32_t>(std::round(srvBox.height() / res));
	Image    out;
	out.width  = W;
	out.height = H;
	out.rgba.assign(size_t(W) * H * 4, 0);
	// 1024 px requests in parallel: several moderate requests beat one big one on these servers
	uint32_t const tile = static_cast<uint32_t>(std::min(1024, s.maxPixels));
	uint32_t       nx = (W + tile - 1) / tile, ny = (H + tile - 1) / tile;
	std::atomic<bool>     any{false};
	std::atomic<uint32_t> next{0};
	std::mutex            errMutex;
	auto                  worker = [&] {
        for (uint32_t k = next.fetch_add(1); k < nx * ny; k = next.fetch_add(1))
        {
            uint32_t tx = k % nx, ty = k / nx;
            uint32_t px0 = tx * tile, py0 = ty * tile;        // py from the top (north)
            uint32_t w = std::min(tile, W - px0), h = std::min(tile, H - py0);
            BBox     b{srvBox.xmin + px0 * res, srvBox.ymax - (py0 + h) * res, srvBox.xmin + (px0 + w) * res, srvBox.ymax - py0 * res};
            auto     resp = http::get(wmsUrl(s.base, s.layer, s.epsg, b, w, h, s.format, s.wms13), true, 120);
            if (!resp.ok())
            {
                std::lock_guard lock(errMutex);
                if (error)
                    *error = resp.error.empty() ? std::format("HTTP {}", resp.status) : resp.error;
                continue;
            }
            Image part = decodeImage(resp.body);
            if (part.empty() || part.width != w || part.height != h)
            {
                std::lock_guard lock(errMutex);
                if (error)
                    *error = std::string(s.name) + " : réponse WMS non décodable";
                continue;
            }
            // disjoint pixel rectangles: no lock needed for the copy
            for (uint32_t row = 0; row < h; ++row)
                std::copy_n(&part.rgba[size_t(row) * w * 4], size_t(w) * 4, &out.rgba[(size_t(py0 + row) * W + px0) * 4]);
            any.store(true);
        }
	};
	{
		std::vector<std::jthread> threads;
		for (unsigned t = 0; t < std::min<unsigned>(6, std::max(1u, nx * ny)); ++t)
			threads.emplace_back(worker);
	}
	return any.load() ? out : Image{};
}

bool mostlyBlank(Image const &img)
{
	// a WMS answering outside its coverage returns white / transparent tiles
	size_t n = img.rgba.size() / 4, white = 0;
	for (size_t k = 0; k < n; k += 97)
		if ((img.rgba[4 * k] > 245 && img.rgba[4 * k + 1] > 245 && img.rgba[4 * k + 2] > 245) || img.rgba[4 * k + 3] < 10)
			++white;
	return white * 97 > n * 9 / 10;
}

}        // namespace

Image fetchOrthophoto(BBox const &box, int epsg, uint32_t pixels, std::string *error, std::string *attribution)
{
	double   longest = std::max(box.width(), box.height());
	double   res     = longest / pixels;        // m / px in the target grid
	uint32_t W       = static_cast<uint32_t>(std::round(box.width() / res));
	uint32_t H       = static_cast<uint32_t>(std::round(box.height() / res));
	BBox     ll      = transformBox(box, epsg, EPSG_WGS84);
	LonLat   centre{0.5 * (ll.xmin + ll.xmax), 0.5 * (ll.ymin + ll.ymax)};

	// regional services first, Sentinel-2 as the universal fallback
	std::vector<OrthoService const *> candidates;
	for (auto const &s : kOrthoServices)
		if (s.extentLonLat.contains(centre.lon, centre.lat))
			candidates.push_back(&s);
	for (auto const *s : candidates)
	{
		std::string err;
		if (s->epsg == epsg)
		{
			// same grid: request the target box directly
			Image img = fetchMosaic(*s, box, res, &err);
			if (!img.empty() && !mostlyBlank(img))
			{
				if (attribution)
					*attribution = s->attribution;
				return img;
			}
		}
		else
		{
			// request in the service CRS (hull + margin), then reproject onto the target grid
			BBox   srv  = transformBox(box, epsg, s->epsg);
			double m    = 4.0 * res;
			srv         = {srv.xmin - m, srv.ymin - m, srv.xmax + m, srv.ymax + m};
			double sres = std::max(srv.width(), srv.height()) / (pixels + 8);
			Image  mosaic = fetchMosaic(*s, srv, sres, &err);
			if (mosaic.empty() || mostlyBlank(mosaic))
			{
				if (error && !err.empty())
					*error = err;
				continue;
			}
			std::vector<uint8_t> bytes;
			auto                 append = [](void *ctx, void *data, int size) {
                auto *v = static_cast<std::vector<uint8_t> *>(ctx);
                v->insert(v->end(), static_cast<uint8_t *>(data), static_cast<uint8_t *>(data) + size);
			};
			if (!stbi_write_png_to_func(append, &bytes, int(mosaic.width), int(mosaic.height), 4, mosaic.rgba.data(), int(mosaic.width) * 4))
				continue;
			auto rgba = warpImage(bytes, srv, s->epsg, box, epsg, W, H, &err);
			if (!rgba.empty())
			{
				if (attribution)
					*attribution = s->attribution;
				Image out;
				out.width  = W;
				out.height = H;
				out.rgba   = std::move(rgba);
				return out;
			}
		}
		if (error && !err.empty())
			*error = err;
	}
	return {};
}

}        // namespace swe::geo
