#pragma once
// "Carte" panel: slippy map (OSM / SPW orthophoto tiles), area selection, data source probing and the
// asynchronous geographic import that produces a case.
#include "geo/Carve.h"
#include "geo/Crs.h"
#include "geo/LocalLibrary.h"
#include "geo/Progress.h"
#include "geo/TerrainBuilder.h"
#include "geo/Tiles.h"
#include "vk/Texture.h"

#include <imgui.h>
#include <vulkan/vulkan.h>

#include <deque>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace swe {

namespace vkc { class Context; }

// Finished import handed to the application
struct ImportOutcome
{
	geo::ImportRequest request;
	geo::ImportResult  result;
	geo::Image         orthophoto;        // empty when not requested / failed
	std::string        error;
};

// Bed edit requested from the map (applied by the application to the loaded case)
struct BedEdit
{
	enum class Type
	{
		Channel,
		Polygon,
	};
	Type                 type = Type::Channel;
	std::vector<geo::XY> points;        // Lambert 72
	double               width = 20.0, depth = 2.0, bankSmoothing = 1.0;        // channel
	double               dz = 1.0;                                             // polygon: lowering (m)
};

class MapPanel
{
  public:
	MapPanel(vkc::Context const &ctx, geo::TerrainBuilder &builder);
	~MapPanel();

	// ImGui window; `frame` = monotonically increasing frame counter (texture lifetime)
	void draw(bool *open, uint64_t frame);
	// Same content inside the current window (collapsing section): toolbar, map of `mapHeight` px, controls below
	void drawEmbedded(uint64_t frame, float mapHeight);

	// An import finished since the last call (consumed once)
	std::optional<ImportOutcome> takeOutcome();
	// A bed edit was requested (consumed once)
	std::optional<BedEdit> takeBedEdit();
	// Geometry of the loaded case (bed edits are only possible on georeferenced cases)
	void setCaseGeometry(std::optional<geo::GridGeometry> const &g) { caseGeo_ = g; }

	// View state persistence
	geo::LonLat center() const { return center_; }
	double      zoom() const { return zoom_; }
	void        setView(geo::LonLat c, double z);
	int         layer() const { return layer_; }
	void        setLayer(int l) { layer_ = l; }

	// Shows a Lambert 72 box on the map (e.g. the loaded case) and frames it
	void showBox(geo::BBox const &boxL72, bool frameIt);

	bool busy() const { return job_.valid(); }

  private:
	struct TileGpu
	{
		vkc::Texture texture;
		VkDescriptorSet descriptor = VK_NULL_HANDLE;
	};

	// map
	void   drawToolbar();
	void   drawMap(uint64_t frame);
	void   drawSidebar();
	void   drawSourcesTable();
	void   drawLibrary();
	void   drawBedTools();
	void   collectGarbage(uint64_t frame);
	ImTextureID tileTexture(geo::TileKey const &key, uint64_t frame);
	geo::LonLat screenToLonLat(ImVec2 p) const;
	ImVec2      lonLatToScreen(geo::LonLat ll) const;
	ImVec2      lambertToScreen(double x, double y) const;
	void        frameBox(geo::BBox const &boxL72);
	void        startProbe();
	void        startImport();
	void        pollJob();

	vkc::Context const  &ctx_;
	geo::TerrainBuilder &builder_;
	geo::TileProvider    tiles_;

	// view
	geo::LonLat center_{5.57, 50.63};        // Liège
	double      zoom_ = 12.0;
	int         layer_ = 0;
	ImVec2      mapPos_{}, mapSize_{};        // screen rectangle of the map canvas (this frame)
	bool        dragging_ = false;
	bool        drawingBox_ = false;
	geo::XY     boxStart_{};

	// selection (Lambert 72)
	std::optional<geo::BBox> box_;
	enum class Tool
	{
		None,
		Box,            // rectangle to import
		Channel,        // river axis polyline
		Polygon,        // area to lower
	};
	Tool                     tool_ = Tool::None;
	std::vector<geo::XY>     sketch_;               // channel / polygon vertices
	BedEdit                  edit_;                 // current parameters
	std::optional<BedEdit>   pendingEdit_;
	std::optional<geo::GridGeometry> caseGeo_;
	char                     locationText_[256] = "";
	std::string              locationError_;

	// import options
	int    cellIndex_   = 2;        // index in kCells
	char   caseName_[64] = "zone_importee";
	int    initialMode_ = 0;        // 0 dry, 1 level everywhere, 2 level in the box
	double level_       = 0.0;
	bool   mergeBathy_  = true;
	bool   fetchOrtho_  = true;
	int    orthoPixels_ = 2048;
	float  manning_     = 0.03f;

	// async job
	geo::Progress                  progress_;
	std::future<ImportOutcome>     job_;
	bool                           jobIsProbe_ = false;
	std::vector<geo::SourceReport> reports_;
	std::optional<ImportOutcome>   outcome_;
	std::string                    status_;

	// library UI
	char libraryFolder_[512] = "";
	bool libraryBusy_ = false;

	// GPU tiles
	std::unordered_map<geo::TileKey, TileGpu, geo::TileKeyHash> gpu_;
	std::deque<std::pair<uint64_t, TileGpu>>                    garbage_;
	std::unordered_map<geo::TileKey, uint64_t, geo::TileKeyHash> lastUsed_;
};

}        // namespace swe
