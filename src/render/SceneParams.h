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
	uint32_t  flags = 0, color_mode = 0, _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(SceneParams) == 144);

}        // namespace swe
