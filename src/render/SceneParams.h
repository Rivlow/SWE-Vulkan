#pragma once
// Host mirror of shaders/render/scene.slang - keep both in sync.
#include <glm/glm.hpp>

#include <cstdint>

namespace swe {

enum SceneFlags : uint32_t
{
	SCENE_LANDCOVER = 1,
	SCENE_OUTSIDE   = 2,
};

struct SceneParams        // std140 uniform buffer
{
	glm::mat4 view_proj{1.0f};
	glm::vec4 cam_pos{0.0f};
	glm::vec4 sun{0.0f};
	float     S = 1, z_scale = 1, depth_ref = 1, water_clarity = 1;
	float     z_base = 0, time = 0, Lx = 0, Ly = 0;
	uint32_t  flags = 0, color_mode = 0;
	float     viewport_w = 1, viewport_h = 1;
	float     cmap_min = 0, cmap_max = 1, cmap_opacity = 0.9f, _pad0 = 0;
};
static_assert(sizeof(SceneParams) == 160);

enum class ColorMode : uint32_t
{
	Realistic = 0,
	Depth     = 1,        // h (m)
	Speed     = 2,        // |u| (m/s)
	Froude    = 3,        // |u| / sqrt(g h)
	Elevation = 4,        // free surface eta (m, relative to z_ref)
};

// Same polynomial as shaders/common/colormap.slang (legend)
glm::vec3 turbo(float x);

}        // namespace swe
