#pragma once
// Host mirror of shaders/render/scene.slang - keep both in sync.
#include <glm/glm.hpp>

#include <cstdint>

namespace swe {

enum SceneFlags : uint32_t
{
	SCENE_LANDCOVER = 1,
	SCENE_OUTSIDE   = 2,
	SCENE_ORTHO     = 4,
	SCENE_THIN_WATER = 8,
	SCENE_ZONES     = 16,
	SCENE_GW_CONTOURS = 32,        // water table contour lines on the ground
	SCENE_CROP        = 64,        // cut box (crop)
	SCENE_GW_XRAY     = 128,       // water table surface seen through the ground
	SCENE_SELECTION   = 256,       // the selection mask (set 2, binding 3) is meaningful
	SCENE_BORDERS     = 512,       // draw the boundary conditions: open sides of the domain and face conditions
	SCENE_MESH        = 1024,      // draw the cell boundaries ("Maillage"), per pixel
	SCENE_BLOCKS      = 2048,      // raw cells: flat-topped columns, no interpolation
};

// Highlight codes (SceneParams::sel.w): a side of the domain, or a group of face conditions
constexpr uint32_t HIGHLIGHT_NONE  = 0;
constexpr uint32_t HIGHLIGHT_SIDE  = 1;         // + side (0..3)
constexpr uint32_t HIGHLIGHT_GROUP = 16;        // + group index

struct SceneParams        // std140 uniform buffer
{
	glm::mat4 view_proj{1.0f};
	glm::vec4 cam_pos{0.0f};
	glm::vec4 sun{0.0f};
	float     S = 1, z_scale = 1, depth_ref = 1, water_clarity = 1;
	float     z_base = 0, time = 0, Lx = 0, Ly = 0;
	uint32_t  flags = 0, color_mode = 0;
	float     viewport_w = 1, viewport_h = 1;
	float     cmap_min = 0, cmap_max = 1, cmap_opacity = 0.9f, ortho_mix = 0;
	glm::uvec4 sel{~0u, ~0u, 0, 0};    // hovered cell i, j (~0 = none), unused, highlight code
	glm::vec4  gw{1, 0.8f, 0, 0};      // contour step of the water table (m), contour opacity, focal length (px)
	glm::uvec4 crop{0, 0, 0, 0};       // cut box i0, j0, i1, j1 inclusive (SCENE_CROP)
};
static_assert(sizeof(SceneParams) == 208);

enum class ColorMode : uint32_t
{
	Realistic = 0,
	Depth     = 1,        // h (m)
	Speed     = 2,        // |u| (m/s)
	Froude    = 3,        // |u| / sqrt(g h)
	Elevation = 4,        // free surface eta (m, relative to z_ref)
	// groundwater, shown on the ground (scene.slang COLOR_GW_*)
	GwTableDepth  = 5,        // zb - H (m)
	GwSaturation  = 6,        // h_gw / (zb - z_sub)
	GwExchange    = 7,        // exchange rate (mm/h, > 0 = infiltration)
	GwInfiltrated = 8,        // net infiltrated depth since t = 0 (mm)
	GwHead        = 9,        // water table elevation (m, relative to z_ref)
	Rain          = 10,       // cumulated rain depth (mm), chronology cases
	Count,
};

// Same polynomial as shaders/common/colormap.slang (legend)
glm::vec3 turbo(float x);
// Diverging map of the signed quantities, same colours as scene.slang diverging()
glm::vec3 diverging(float x);

// Colour of a boundary condition kind (BcKind), same as scene.slang bc_color(): shared by the 3D view and the GUI
glm::vec3 bcColor(uint32_t kind);
// Colour of the cell selection
constexpr glm::vec3 kSelectionColor{1.0f, 0.80f, 0.20f};

}        // namespace swe
