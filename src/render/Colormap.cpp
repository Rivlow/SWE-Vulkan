#include "render/SceneParams.h"

#include <algorithm>

namespace swe {

// Turbo colour map (Google, polynomial approximation), same coefficients as shaders/common/colormap.slang
glm::vec3 turbo(float x)
{
	glm::vec4 const kRed4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f);
	glm::vec4 const kGreen4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f);
	glm::vec4 const kBlue4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f);
	glm::vec2 const kRed2(-152.94239396f, 59.28637943f);
	glm::vec2 const kGreen2(4.27729857f, 2.82956604f);
	glm::vec2 const kBlue2(-89.90310912f, 27.34824973f);
	x = std::clamp(x, 0.0f, 1.0f);
	glm::vec4 v4(1.0f, x, x * x, x * x * x);
	glm::vec2 v2 = glm::vec2(v4.z, v4.w) * v4.z;
	return glm::clamp(glm::vec3(glm::dot(v4, kRed4) + glm::dot(v2, kRed2), glm::dot(v4, kGreen4) + glm::dot(v2, kGreen2),
	                            glm::dot(v4, kBlue4) + glm::dot(v2, kBlue2)),
	                  0.0f, 1.0f);
}

}        // namespace swe
