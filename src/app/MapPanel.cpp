#include "app/MapPanel.h"

#include "util/Log.h"
#include "vk/Context.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace swe {

namespace {

constexpr double kCells[]      = {0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0};
constexpr char const *kCellLabels[] = {"0,5 m", "1 m", "2 m", "5 m", "10 m", "25 m", "50 m"};
constexpr int    kMinZoom = 4, kMaxZoom = 19;
constexpr double kTilePx  = 256.0;
constexpr size_t kMaxCells = 16'000'000;

std::string humanCells(size_t n)
{
	return n >= 1'000'000 ? std::format("{:.2f} M", n * 1e-6) : n >= 1000 ? std::format("{:.0f} k", n * 1e-3) : std::to_string(n);
}

std::string humanBytes(double b)
{
	return b >= 1e9 ? std::format("{:.2f} Go", b * 1e-9) : std::format("{:.0f} Mo", b * 1e-6);
}

ImTextureID toTextureId(VkDescriptorSet ds)
{
	return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(ds));
}

}        // namespace

MapPanel::MapPanel(vkc::Context const &ctx, geo::TerrainBuilder &builder) :
    ctx_(ctx), builder_(builder), tiles_({geo::osmLayer(), geo::sentinelLayer(), geo::spwOrthoLayer()}, 4)
{
	auto const &folders = builder_.library().folders();
	if (!folders.empty())
		std::snprintf(libraryFolder_, sizeof(libraryFolder_), "%s", folders.back().string().c_str());
}

MapPanel::~MapPanel()
{
	if (job_.valid())
	{
		progress_.cancel();
		job_.wait();
	}
	ctx_.device.waitIdle();
	for (auto &[key, t] : gpu_)
		if (t.descriptor)
			ImGui_ImplVulkan_RemoveTexture(t.descriptor);
	for (auto &[frame, t] : garbage_)
		if (t.descriptor)
			ImGui_ImplVulkan_RemoveTexture(t.descriptor);
}

void MapPanel::setView(geo::LonLat c, double z)
{
	center_ = c;
	zoom_   = std::clamp(z, double(kMinZoom), double(kMaxZoom));
}

// ---------------------------------------------------------------------------------------------
// Projection helpers (screen <-> world), Web Mercator at fractional zoom
// ---------------------------------------------------------------------------------------------

geo::LonLat MapPanel::screenToLonLat(ImVec2 p) const
{
	double scale = kTilePx * std::ldexp(1.0, 0) * std::pow(2.0, zoom_);        // px per world (tile units at z = 0)
	auto   c     = geo::toTile(center_, 0);
	double tx    = c.x + (p.x - (mapPos_.x + mapSize_.x * 0.5)) / scale;
	double ty    = c.y + (p.y - (mapPos_.y + mapSize_.y * 0.5)) / scale;
	return geo::fromTile({tx, ty}, 0);
}

ImVec2 MapPanel::lonLatToScreen(geo::LonLat ll) const
{
	double scale = kTilePx * std::pow(2.0, zoom_);
	auto   c     = geo::toTile(center_, 0);
	auto   t     = geo::toTile(ll, 0);
	return ImVec2(float(mapPos_.x + mapSize_.x * 0.5 + (t.x - c.x) * scale), float(mapPos_.y + mapSize_.y * 0.5 + (t.y - c.y) * scale));
}

ImVec2 MapPanel::projToScreen(double x, double y, int epsg) const
{
	geo::transform(epsg, geo::EPSG_WGS84, x, y);
	return lonLatToScreen({x, y});
}

int MapPanel::sketchEpsg() const
{
	return caseGeo_ ? caseGeo_->epsg : boxEpsg_;
}

void MapPanel::frameBox(geo::BBox const &box, int epsg)
{
	auto ll  = geo::transformBox(box, epsg, geo::EPSG_WGS84);
	center_  = {0.5 * (ll.xmin + ll.xmax), 0.5 * (ll.ymin + ll.ymax)};
	auto a   = geo::toTile({ll.xmin, ll.ymax}, 0), b = geo::toTile({ll.xmax, ll.ymin}, 0);
	double w = std::max(b.x - a.x, 1e-12), h = std::max(b.y - a.y, 1e-12);
	double sx = std::max(mapSize_.x, 200.0f) * 0.8 / (kTilePx * w), sy = std::max(mapSize_.y, 200.0f) * 0.8 / (kTilePx * h);
	zoom_     = std::clamp(std::log2(std::min(sx, sy)), double(kMinZoom), double(kMaxZoom));
}

void MapPanel::showBox(geo::BBox const &box, int epsg, bool frameIt)
{
	box_     = box;
	boxEpsg_ = epsg;
	if (frameIt)
		frameBox(box, epsg);
}

// ---------------------------------------------------------------------------------------------
// GPU tiles
// ---------------------------------------------------------------------------------------------

ImTextureID MapPanel::tileTexture(geo::TileKey const &key, uint64_t frame)
{
	auto it = gpu_.find(key);
	if (it != gpu_.end())
	{
		lastUsed_[key] = frame;
		return toTextureId(it->second.descriptor);
	}
	geo::Image const *img = nullptr;
	if (tiles_.get(key, &img) != geo::TileState::Ready || !img)
		return 0;
	TileGpu t;
	t.texture    = vkc::Texture(ctx_, img->width, img->height, img->rgba.data(), false);
	t.descriptor = ImGui_ImplVulkan_AddTexture(t.texture.sampler(), t.texture.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	auto id      = toTextureId(t.descriptor);
	gpu_.emplace(key, std::move(t));
	lastUsed_[key] = frame;
	return id;
}

void MapPanel::collectGarbage(uint64_t frame)
{
	// textures not drawn for a while -> deferred destruction (the GPU may still read the last frames)
	if (gpu_.size() > 400)
		for (auto it = gpu_.begin(); it != gpu_.end();)
		{
			auto used = lastUsed_[it->first];
			if (frame - used > 120)
			{
				garbage_.emplace_back(frame, std::move(it->second));
				lastUsed_.erase(it->first);
				it = gpu_.erase(it);
			}
			else
				++it;
		}
	while (!garbage_.empty() && frame - garbage_.front().first > 4)
	{
		if (garbage_.front().second.descriptor)
			ImGui_ImplVulkan_RemoveTexture(garbage_.front().second.descriptor);
		garbage_.pop_front();
	}
}

// ---------------------------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------------------------

void MapPanel::startProbe()
{
	if (!box_ || job_.valid())
		return;
	progress_.reset();
	jobIsProbe_ = true;
	status_     = "Sondage des sources…";
	auto box    = *box_;
	int  epsg   = boxEpsg_;
	job_        = std::async(std::launch::async, [this, box, epsg] {
        ImportOutcome o;
        o.request.box  = box;
        o.request.epsg = epsg;
        reports_       = {};
        auto rep       = builder_.probe(box, epsg, progress_, true);
        {
            // hand over through the outcome (consumed on the GUI thread)
            o.result.sources = rep;
        }
        return o;
    });
}

void MapPanel::startImport()
{
	if (!box_ || job_.valid())
		return;
	progress_.reset();
	jobIsProbe_ = false;
	status_     = "Import…";
	geo::ImportRequest req;
	req.box        = *box_;
	req.epsg       = boxEpsg_;
	req.cell       = kCells[cellIndex_];
	req.mergeBathy = mergeBathy_;
	req.name       = caseName_;
	req.label      = caseName_;
	req.initial    = initialMode_ == 0 ? geo::InitialWater::Dry : initialMode_ == 1 ? geo::InitialWater::Level : geo::InitialWater::LevelInRectangle;
	req.level      = level_;
	req.levelBox   = *box_;
	req.manning    = manning_;
	// display hints from the size of the area
	double span    = std::max(req.box.width(), req.box.height());
	req.depthRef   = span > 5000 ? 5.0 : span > 1000 ? 4.0 : 2.0;
	req.zScale     = span > 5000 ? 5.0 : span > 1000 ? 4.0 : 2.0;
	bool ortho     = fetchOrtho_;
	int  orthoPx   = orthoPixels_;
	job_           = std::async(std::launch::async, [this, req, ortho, orthoPx] {
        ImportOutcome o;
        o.request = req;
        o.result  = builder_.build(req, progress_, &o.error);
        if (!o.result.bed.v.empty() && ortho && !progress_.cancelled())
        {
            progress_.set(0.95f, "Orthophoto…");
            std::string err, attribution;
            o.orthophoto = geo::fetchOrthophoto(req.box, req.epsg, static_cast<uint32_t>(orthoPx), &err, &attribution);
            if (o.orthophoto.empty())
                log::warn("orthophoto: {}", err);
            else if (!attribution.empty())
                o.result.attributions.push_back(attribution);
        }
        return o;
    });
}

void MapPanel::pollJob()
{
	if (!job_.valid() || job_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
		return;
	auto o = job_.get();
	if (jobIsProbe_)
	{
		reports_ = o.result.sources;
		status_  = progress_.cancelled() ? "Sondage annulé" : "Sources sondées";
	}
	else if (progress_.cancelled())
		status_ = "Import annulé";
	else if (o.result.bed.v.empty())
		status_ = "Échec : " + o.error;
	else
	{
		status_  = std::format("Import terminé : {} x {} mailles", o.result.bed.nx, o.result.bed.ny);
		reports_ = o.result.sources;
		outcome_ = std::move(o);
	}
}

std::optional<ImportOutcome> MapPanel::takeOutcome()
{
	auto o = std::move(outcome_);
	outcome_.reset();
	return o;
}

std::optional<BedEdit> MapPanel::takeBedEdit()
{
	auto e = std::move(pendingEdit_);
	pendingEdit_.reset();
	return e;
}

// ---------------------------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------------------------

void MapPanel::draw(bool *open, uint64_t frame)
{
	pollJob();
	collectGarbage(frame);
	ImGui::SetNextWindowSize(ImVec2(1000, 620), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Carte", open, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
	{
		ImGui::End();
		return;
	}
	drawToolbar();
	float sidebar = 340.0f;
	ImGui::BeginChild("map", ImVec2(ImGui::GetContentRegionAvail().x - sidebar - ImGui::GetStyle().ItemSpacing.x, 0), ImGuiChildFlags_Borders,
	                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	drawMap(frame);
	ImGui::EndChild();
	ImGui::SameLine();
	ImGui::BeginChild("sidebar", ImVec2(sidebar, 0));
	drawSidebar();
	ImGui::EndChild();
	ImGui::End();
}

void MapPanel::drawEmbedded(uint64_t frame, float mapHeight)
{
	pollJob();
	collectGarbage(frame);
	ImGui::PushID("map-embedded");
	drawToolbar();
	ImGui::BeginChild("map", ImVec2(0, mapHeight), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
	drawMap(frame);
	ImGui::EndChild();
	drawSidebar();
	ImGui::PopID();
}

void MapPanel::drawToolbar()
{
	auto const &layers = tiles_.layers();
	bool        narrow = ImGui::GetContentRegionAvail().x < 760.0f;
	ImGui::SetNextItemWidth(narrow ? -1.0f : 220.0f);
	if (ImGui::BeginCombo("##layer", layers[static_cast<size_t>(layer_)].name.c_str()))
	{
		for (int k = 0; k < int(layers.size()); ++k)
			if (ImGui::Selectable(layers[static_cast<size_t>(k)].name.c_str(), k == layer_))
				layer_ = k;
		ImGui::EndCombo();
	}
	if (!narrow)
		ImGui::SameLine();
	ImGui::SetNextItemWidth(narrow ? -60.0f : 360.0f);
	bool go = ImGui::InputTextWithHint("##loc", "lat, lon | lien Google Maps | x, y L72", locationText_, sizeof(locationText_),
	                                   ImGuiInputTextFlags_EnterReturnsTrue);
	ImGui::SameLine();
	go |= ImGui::Button("Aller");
	if (narrow)
		ImGui::NewLine(), ImGui::SameLine(0, 0);
	else
		ImGui::SameLine();
	if (go)
	{
		geo::LonLat ll;
		if (geo::parseLocation(locationText_, ll))
		{
			center_        = ll;
			zoom_          = std::max(zoom_, 14.0);
			locationError_ = {};
		}
		else
			locationError_ = "Position non reconnue";
	}
	if (ImGui::Button(tool_ == Tool::Box ? "Zone : cliquer-glisser sur la carte" : "Tracer une zone"))
		tool_ = tool_ == Tool::Box ? Tool::None : Tool::Box;
	if (box_)
	{
		ImGui::SameLine();
		if (ImGui::Button("Cadrer"))
			frameBox(*box_, boxEpsg_);
		ImGui::SameLine();
		if (ImGui::Button("Effacer"))
		{
			box_.reset();
			reports_.clear();
		}
	}
	if (!locationError_.empty())
	{
		ImGui::SameLine();
		ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "%s", locationError_.c_str());
	}
}

void MapPanel::drawMap(uint64_t frame)
{
	mapPos_  = ImGui::GetCursorScreenPos();
	mapSize_ = ImGui::GetContentRegionAvail();
	if (mapSize_.x < 50 || mapSize_.y < 50)
		return;
	auto *dl = ImGui::GetWindowDrawList();
	dl->PushClipRect(mapPos_, ImVec2(mapPos_.x + mapSize_.x, mapPos_.y + mapSize_.y), true);
	dl->AddRectFilled(mapPos_, ImVec2(mapPos_.x + mapSize_.x, mapPos_.y + mapSize_.y), IM_COL32(220, 226, 232, 255));

	// --- tiles at the integer zoom level, scaled by the fractional part
	int    zi     = static_cast<int>(std::floor(zoom_));
	double scaleZ = std::pow(2.0, zoom_ - zi);        // 1..2
	double tilePx = kTilePx * scaleZ;
	auto   c      = geo::toTile(center_, zi);
	double n      = std::ldexp(1.0, zi);
	// tile range covering the canvas
	double x0 = c.x - mapSize_.x * 0.5 / tilePx, x1 = c.x + mapSize_.x * 0.5 / tilePx;
	double y0 = c.y - mapSize_.y * 0.5 / tilePx, y1 = c.y + mapSize_.y * 0.5 / tilePx;
	auto const &layer = tiles_.layers()[static_cast<size_t>(layer_)];
	int  missing = 0;
	for (int ty = static_cast<int>(std::floor(y0)); ty <= static_cast<int>(std::floor(y1)); ++ty)
	{
		if (ty < 0 || ty >= int(n))
			continue;
		for (int tx = static_cast<int>(std::floor(x0)); tx <= static_cast<int>(std::floor(x1)); ++tx)
		{
			int wx = ((tx % int(n)) + int(n)) % int(n);        // wrap around the antimeridian
			ImVec2 a(float(mapPos_.x + mapSize_.x * 0.5 + (tx - c.x) * tilePx), float(mapPos_.y + mapSize_.y * 0.5 + (ty - c.y) * tilePx));
			ImVec2 b(a.x + float(tilePx) + 0.5f, a.y + float(tilePx) + 0.5f);
			if (zi < layer.minZoom || zi > layer.maxZoom)
				continue;
			ImTextureID tex = tileTexture({zi, wx, ty, layer_}, frame);
			if (tex)
				dl->AddImage(tex, a, b);
			else
			{
				// fallback: parent tile (zoomed in) while loading
				bool drawn = false;
				for (int up = 1; up <= 3 && zi - up >= layer.minZoom && !drawn; ++up)
				{
					int         pz = zi - up, px = wx >> up, py = ty >> up;
					ImTextureID ptex = tileTexture({pz, px, py, layer_}, frame);
					if (!ptex)
						continue;
					double f  = std::ldexp(1.0, up);
					double u0 = (wx - px * f) / f, v0 = (ty - py * f) / f;
					dl->AddImage(ptex, a, b, ImVec2(float(u0), float(v0)), ImVec2(float(u0 + 1.0 / f), float(v0 + 1.0 / f)));
					drawn = true;
				}
				++missing;
			}
		}
	}

	// --- selection rectangle (projected box: draw its 4 corners)
	auto drawBox = [&](geo::BBox const &bx, int epsg, ImU32 fill, ImU32 line) {
		ImVec2 p[4] = {projToScreen(bx.xmin, bx.ymin, epsg), projToScreen(bx.xmax, bx.ymin, epsg), projToScreen(bx.xmax, bx.ymax, epsg),
		               projToScreen(bx.xmin, bx.ymax, epsg)};
		dl->AddConvexPolyFilled(p, 4, fill);
		dl->AddPolyline(p, 4, line, ImDrawFlags_Closed, 2.0f);
	};
	if (box_)
		drawBox(*box_, boxEpsg_, IM_COL32(255, 160, 40, 45), IM_COL32(255, 140, 20, 255));
	// --- bed edit sketch (river axis with its width, or polygon)
	if (!sketch_.empty())
	{
		std::vector<ImVec2> pts;
		for (auto const &p : sketch_)
			pts.push_back(projToScreen(p.x, p.y, sketchEpsg()));
		double mpp = 156543.03392 * std::cos(center_.lat * 3.14159265 / 180.0) / std::pow(2.0, zoom_);
		bool   channel = tool_ == Tool::Channel || edit_.type == BedEdit::Type::Channel;
		if (channel)
		{
			if (pts.size() >= 2)
				dl->AddPolyline(pts.data(), int(pts.size()), IM_COL32(40, 120, 255, 110), ImDrawFlags_None, float(edit_.width / mpp));
			if (pts.size() >= 2)
				dl->AddPolyline(pts.data(), int(pts.size()), IM_COL32(20, 80, 255, 255), ImDrawFlags_None, 2.0f);
		}
		else if (pts.size() >= 3)
		{
			dl->AddConvexPolyFilled(pts.data(), int(pts.size()), IM_COL32(200, 60, 255, 40));
			dl->AddPolyline(pts.data(), int(pts.size()), IM_COL32(180, 40, 255, 255), ImDrawFlags_Closed, 2.0f);
		}
		else if (pts.size() == 2)
			dl->AddLine(pts[0], pts[1], IM_COL32(180, 40, 255, 255), 2.0f);
		for (auto const &p : pts)
			dl->AddCircleFilled(p, 4.0f, IM_COL32(255, 255, 255, 255));
	}

	// --- input
	ImGui::InvisibleButton("canvas", mapSize_, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
	bool hovered = ImGui::IsItemHovered();
	auto &io     = ImGui::GetIO();
	if (hovered)
		ImGui::SetItemKeyOwner(ImGuiKey_MouseWheelY);        // the wheel zooms the map, it must not scroll the panel
	if (hovered && io.MouseWheel != 0.0f)
	{
		// zoom around the cursor
		auto   before = screenToLonLat(io.MousePos);
		zoom_         = std::clamp(zoom_ + io.MouseWheel * 0.5, double(kMinZoom), double(kMaxZoom));
		auto   after  = screenToLonLat(io.MousePos);
		auto   tb = geo::toTile(before, 0), ta = geo::toTile(after, 0), tc = geo::toTile(center_, 0);
		center_       = geo::fromTile({tc.x + (tb.x - ta.x), tc.y + (tb.y - ta.y)}, 0);
		tiles_.cancelPending();
	}
	bool sketching = tool_ == Tool::Channel || tool_ == Tool::Polygon;
	// sketch tools: left click adds a vertex, right click removes the last one, double click finishes
	if (sketching && hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !ImGui::IsMouseDragging(ImGuiMouseButton_Left, 4.0f))
	{
		if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			tool_ = Tool::None;
		else
		{
			auto   ll = screenToLonLat(io.MousePos);
			double x = ll.lon, y = ll.lat;
			geo::transform(geo::EPSG_WGS84, sketchEpsg(), x, y);
			sketch_.push_back({x, y});
		}
	}
	if (sketching && hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !sketch_.empty())
		sketch_.pop_back();
	bool boxMode    = tool_ == Tool::Box;
	bool drawingNow = boxMode || (!sketching && ImGui::IsMouseDown(ImGuiMouseButton_Right));
	if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && !boxMode)
	{
		// pan
		double scale = kTilePx * std::pow(2.0, zoom_);
		auto   tc    = geo::toTile(center_, 0);
		center_      = geo::fromTile({tc.x - io.MouseDelta.x / scale, tc.y - io.MouseDelta.y / scale}, 0);
	}
	// rectangle: left drag in box mode, or right drag anytime (outside the sketch tools)
	int drawButton = boxMode ? ImGuiMouseButton_Left : ImGuiMouseButton_Right;
	if (hovered && ImGui::IsMouseClicked(drawButton) && drawingNow)
	{
		// the working projection is chosen for the point where the drag starts
		auto ll     = screenToLonLat(io.MousePos);
		boxEpsg_    = geo::suggestEpsg(ll);
		boxStart_   = {ll.lon, ll.lat};
		geo::transform(geo::EPSG_WGS84, boxEpsg_, boxStart_.x, boxStart_.y);
		drawingBox_ = true;
	}
	if (drawingBox_)
	{
		auto    ll  = screenToLonLat(io.MousePos);
		geo::XY cur{ll.lon, ll.lat};
		geo::transform(geo::EPSG_WGS84, boxEpsg_, cur.x, cur.y);
		geo::BBox b{std::min(boxStart_.x, cur.x), std::min(boxStart_.y, cur.y), std::max(boxStart_.x, cur.x), std::max(boxStart_.y, cur.y)};
		// snap to whole metres
		b = {std::floor(b.xmin), std::floor(b.ymin), std::ceil(b.xmax), std::ceil(b.ymax)};
		if (!b.empty())
			drawBox(b, boxEpsg_, IM_COL32(255, 160, 40, 30), IM_COL32(255, 140, 20, 200));
		if (ImGui::IsMouseReleased(drawButton))
		{
			drawingBox_ = false;
			if (b.width() >= 10.0 && b.height() >= 10.0)
			{
				box_ = b;
				reports_.clear();
				tool_ = Tool::None;
			}
		}
	}

	// --- overlays: cursor position, scale bar, attribution
	if (hovered)
	{
		auto    ll   = screenToLonLat(io.MousePos);
		int     epsg = box_ ? boxEpsg_ : geo::suggestEpsg(ll);
		geo::XY xy{ll.lon, ll.lat};
		geo::transform(geo::EPSG_WGS84, epsg, xy.x, xy.y);
		auto txt = std::format("{:.5f}, {:.5f}   {} {:.0f}, {:.0f}", ll.lat, ll.lon, geo::crsName(epsg), xy.x, xy.y);
		dl->AddRectFilled(ImVec2(mapPos_.x + 6, mapPos_.y + 6), ImVec2(mapPos_.x + 16 + ImGui::CalcTextSize(txt.c_str()).x, mapPos_.y + 26), IM_COL32(0, 0, 0, 140), 3.0f);
		dl->AddText(ImVec2(mapPos_.x + 11, mapPos_.y + 9), IM_COL32_WHITE, txt.c_str());
	}
	{
		// scale bar: metres per pixel at the centre latitude
		double mpp  = 156543.03392 * std::cos(center_.lat * 3.14159265 / 180.0) / std::pow(2.0, zoom_);
		double target = 100.0 * mpp;        // metres for 100 px
		double nice = std::pow(10.0, std::floor(std::log10(target)));
		for (double m : {1.0, 2.0, 5.0})
			if (nice * m <= target)
				target = nice * m;
		float  px = float(target / mpp);
		ImVec2 a(mapPos_.x + 12, mapPos_.y + mapSize_.y - 18);
		dl->AddLine(a, ImVec2(a.x + px, a.y), IM_COL32(20, 20, 20, 255), 3.0f);
		auto label = target >= 1000 ? std::format("{:g} km", target / 1000) : std::format("{:g} m", target);
		dl->AddText(ImVec2(a.x, a.y - 18), IM_COL32(20, 20, 20, 255), label.c_str());
	}
	{
		auto const &attr = layer.attribution;
		ImVec2      ts   = ImGui::CalcTextSize(attr.c_str());
		ImVec2      p(mapPos_.x + mapSize_.x - ts.x - 10, mapPos_.y + mapSize_.y - ts.y - 6);
		dl->AddRectFilled(ImVec2(p.x - 4, p.y - 2), ImVec2(p.x + ts.x + 4, p.y + ts.y + 2), IM_COL32(255, 255, 255, 180), 3.0f);
		dl->AddText(p, IM_COL32(30, 30, 30, 255), attr.c_str());
	}
	if (missing > 0 || tiles_.pendingCount() > 0)
	{
		auto txt = std::format("chargement… {}", tiles_.pendingCount());
		dl->AddText(ImVec2(mapPos_.x + mapSize_.x - 130, mapPos_.y + 8), IM_COL32(40, 40, 40, 255), txt.c_str());
	}
	dl->PopClipRect();
}

void MapPanel::drawSidebar()
{
	auto const &lib = builder_.library();
	ImGui::SeparatorText("Zone");
	if (!box_)
	{
		ImGui::TextWrapped("Tracez un rectangle sur la carte (bouton « Tracer une zone » puis cliquer-glisser, ou glisser avec le bouton droit).");
		ImGui::TextDisabled("Glisser = déplacer, molette = zoom.");
	}
	else
	{
		auto const &b = *box_;
		ImGui::Text("%.0f x %.0f m   (%s)", b.width(), b.height(), geo::crsName(boxEpsg_).c_str());
		ImGui::TextDisabled("%.0f, %.0f -> %.0f, %.0f", b.xmin, b.ymin, b.xmax, b.ymax);
		ImGui::SetNextItemWidth(-90);
		ImGui::Combo("maille", &cellIndex_, kCellLabels, IM_ARRAYSIZE(kCellLabels));
		double cell = kCells[cellIndex_];
		size_t nx = static_cast<size_t>(std::round(b.width() / cell)), ny = static_cast<size_t>(std::round(b.height() / cell));
		size_t cells = nx * ny;
		double gpuBytes = double(cells) * (16.0 * 4 + 4 + 8 + 16 * 2 + 12 * 4);        // Q, Q_old, K(2), zb, S_b, fluxes, effects
		ImGui::Text("%zu x %zu = %s mailles (~%s GPU)", nx, ny, humanCells(cells).c_str(), humanBytes(gpuBytes).c_str());
		if (cells > kMaxCells)
			ImGui::TextColored(ImVec4(1, 0.5f, 0.4f, 1), "Trop de mailles : agrandir la maille ou réduire la zone");
		else if (cells > 4'000'000)
			ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "Grande grille : simulation lente");

		ImGui::SeparatorText("Sources");
		ImGui::BeginDisabled(job_.valid());
		if (ImGui::Button("Sonder les sources"))
			startProbe();
		ImGui::EndDisabled();
		ImGui::SameLine();
		ImGui::TextDisabled("(couverture réelle)");
		drawSourcesTable();

		ImGui::SeparatorText("Cas");
		ImGui::SetNextItemWidth(-90);
		ImGui::InputText("nom", caseName_, sizeof(caseName_));
		ImGui::SetNextItemWidth(-90);
		ImGui::Combo("eau initiale", &initialMode_, "sec\0niveau partout\0niveau dans la zone\0");
		if (initialMode_ != 0)
		{
			ImGui::SetNextItemWidth(-90);
			ImGui::InputDouble("cote (m DNG)", &level_, 0.5, 5.0, "%.2f");
		}
		ImGui::SetNextItemWidth(-90);
		ImGui::SliderFloat("Manning n", &manning_, 0.0f, 0.1f, "%.3f");
		ImGui::Checkbox("Fusionner la bathymétrie", &mergeBathy_);
		ImGui::Checkbox("Orthophoto drapée", &fetchOrtho_);
		if (fetchOrtho_)
		{
			auto ll = geo::transformBox(b, boxEpsg_, geo::EPSG_WGS84);
			ImGui::TextDisabled("%s", geo::orthophotoSourceName({0.5 * (ll.xmin + ll.xmax), 0.5 * (ll.ymin + ll.ymax)}).c_str());
			ImGui::SameLine();
			ImGui::SetNextItemWidth(90);
			ImGui::Combo("##orthopx", &orthoPixels_, "1024\0" "2048\0" "4096\0");
			orthoPixels_ = orthoPixels_ == 0 ? 1024 : orthoPixels_ == 1 ? 2048 : orthoPixels_ == 2 ? 4096 : orthoPixels_;
		}
		ImGui::BeginDisabled(job_.valid() || cells > kMaxCells);
		if (ImGui::Button("Importer et charger", ImVec2(-1, 0)))
			startImport();
		ImGui::EndDisabled();
	}

	if (job_.valid())
	{
		ImGui::ProgressBar(progress_.fraction(), ImVec2(-60, 0), progress_.stage().c_str());
		ImGui::SameLine();
		if (ImGui::Button("Stop"))
			progress_.cancel();
	}
	else if (!status_.empty())
		ImGui::TextWrapped("%s", status_.c_str());

	ImGui::Spacing();
	if (ImGui::CollapsingHeader("Lit de rivière (zones sans bathymétrie)"))
		drawBedTools();
	if (ImGui::CollapsingHeader("Données locales (MNT / bathymétrie SPW)"))
		drawLibrary();
	ImGui::Spacing();
	ImGui::TextDisabled("%zu raster(s) locaux indexés", lib.entries().size());
}

void MapPanel::drawSourcesTable()
{
	if (reports_.empty())
	{
		ImGui::TextDisabled("Priorité : MNT local > Flandre / NL / NRW / France > Copernicus 30 m ;\nla bathymétrie SPW remplace le MNT là où elle existe.");
		return;
	}
	if (ImGui::BeginTable("sources", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
	{
		ImGui::TableSetupColumn("source");
		ImGui::TableSetupColumn("emprise", ImGuiTableColumnFlags_WidthFixed, 60);
		ImGui::TableSetupColumn("données", ImGuiTableColumnFlags_WidthFixed, 60);
		ImGui::TableHeadersRow();
		for (auto const &r : reports_)
		{
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			bool dim = r.footprint <= 0.0;
			if (dim)
				ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
			ImGui::TextUnformatted(r.name.c_str());
			if (!r.error.empty() && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", r.error.c_str());
			ImGui::TableNextColumn();
			ImGui::Text("%.0f %%", 100.0 * r.footprint);
			ImGui::TableNextColumn();
			if (r.data >= 0.0)
				ImGui::Text("%.0f %%%s", 100.0 * r.data, r.used ? " *" : "");
			else
				ImGui::TextUnformatted("-");
			if (dim)
				ImGui::PopStyleColor();
		}
		ImGui::EndTable();
	}
}

void MapPanel::drawBedTools()
{
	ImGui::TextWrapped("Le LiDAR mesure la surface de l'eau : là où aucune bathymétrie n'existe, creusez un lit "
	                   "estimé le long de l'axe de la rivière, ou abaissez une zone.");
	if (!caseGeo_)
	{
		ImGui::TextColored(ImVec4(1, 0.8f, 0.4f, 1), "Chargez d'abord un cas géoréférencé (importé).");
	}
	bool channel = edit_.type == BedEdit::Type::Channel;
	if (ImGui::RadioButton("axe de rivière", channel))
	{
		edit_.type = BedEdit::Type::Channel;
		if (tool_ == Tool::Polygon)
			tool_ = Tool::Channel;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("zone à abaisser", !channel))
	{
		edit_.type = BedEdit::Type::Polygon;
		if (tool_ == Tool::Channel)
			tool_ = Tool::Polygon;
	}
	Tool wanted = edit_.type == BedEdit::Type::Channel ? Tool::Channel : Tool::Polygon;
	if (ImGui::Button(tool_ == wanted ? "Tracé en cours : clic = point, double-clic = fin" : "Tracer sur la carte"))
	{
		if (tool_ == wanted)
			tool_ = Tool::None;
		else
		{
			tool_ = wanted;
			sketch_.clear();
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Effacer le tracé"))
	{
		sketch_.clear();
		tool_ = Tool::None;
	}
	ImGui::Text("%zu point(s)", sketch_.size());
	if (edit_.type == BedEdit::Type::Channel)
	{
		float w = float(edit_.width), d = float(edit_.depth), s = float(edit_.bankSmoothing);
		ImGui::SetNextItemWidth(-110);
		if (ImGui::SliderFloat("largeur (m)", &w, 2.0f, 300.0f, "%.0f", ImGuiSliderFlags_Logarithmic))
			edit_.width = w;
		ImGui::SetNextItemWidth(-110);
		if (ImGui::SliderFloat("profondeur (m)", &d, 0.2f, 15.0f, "%.1f"))
			edit_.depth = d;
		ImGui::SetNextItemWidth(-110);
		if (ImGui::SliderFloat("berges", &s, 0.0f, 1.0f, "%.2f"))
			edit_.bankSmoothing = s;
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("1 = profil parabolique, 0 = fond plat et berges raides");
	}
	else
	{
		float dz = float(edit_.dz);
		ImGui::SetNextItemWidth(-110);
		if (ImGui::SliderFloat("abaissement (m)", &dz, -10.0f, 10.0f, "%.1f"))
			edit_.dz = dz;
	}
	size_t need = edit_.type == BedEdit::Type::Channel ? 2 : 3;
	ImGui::BeginDisabled(!caseGeo_ || sketch_.size() < need);
	if (ImGui::Button(edit_.type == BedEdit::Type::Channel ? "Creuser le lit" : "Appliquer à la zone", ImVec2(-1, 0)))
	{
		BedEdit e   = edit_;
		e.points    = sketch_;
		pendingEdit_ = e;
		tool_        = Tool::None;
	}
	ImGui::EndDisabled();
	ImGui::TextDisabled("Le fond n'est jamais relevé par le creusement ; « Enregistrer le cas »\n(panneau Simulation) garde le résultat.");
}

void MapPanel::drawLibrary()
{
	auto &lib = builder_.library();
	ImGui::TextWrapped("Dossiers de GeoTIFF (ou zips) : MNT LiDAR SPW, bathymétrie SPW… Les fichiers contenant « BATHY » sont pris comme bathymétrie.");
	for (size_t k = 0; k < lib.folders().size(); ++k)
	{
		ImGui::PushID(int(k));
		if (ImGui::SmallButton("x"))
		{
			lib.removeFolder(k);
			ImGui::PopID();
			break;
		}
		ImGui::SameLine();
		ImGui::TextUnformatted(lib.folders()[k].string().c_str());
		ImGui::PopID();
	}
	ImGui::SetNextItemWidth(-70);
	ImGui::InputTextWithHint("##folder", "C:\\...\\dossier", libraryFolder_, sizeof(libraryFolder_));
	ImGui::SameLine();
	if (ImGui::Button("Ajouter") && libraryFolder_[0])
		lib.addFolder(libraryFolder_);
	if (ImGui::Button("Ré-indexer") && !job_.valid())
	{
		lib.rescan();
		builder_.refreshLocal();
		reports_.clear();
	}
	ImGui::SameLine();
	ImGui::TextDisabled("MNT wallon 1 m : zip provincial SPW (voir aide)");
	for (auto &e : lib.entries())
	{
		ImGui::PushID(e.info.path.c_str());
		ImGui::Checkbox("##on", &e.enabled);
		ImGui::SameLine();
		int kind = e.kind == geo::SourceKind::Bathymetry ? 1 : 0;
		ImGui::SetNextItemWidth(80);
		if (ImGui::Combo("##kind", &kind, "MNT\0bathy\0"))
			e.kind = kind == 1 ? geo::SourceKind::Bathymetry : geo::SourceKind::Terrain;
		ImGui::SameLine();
		ImGui::Text("%s  (%.2g m)", e.label.c_str(), e.info.resX);
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s\n%u x %u px, EPSG:%d", e.info.path.c_str(), e.info.width, e.info.height, e.info.epsg);
		ImGui::PopID();
	}
	if (!lib.entries().empty() && ImGui::Button("Enregistrer la bibliothèque"))
	{
		lib.save();
		builder_.refreshLocal();
	}
}

}        // namespace swe
