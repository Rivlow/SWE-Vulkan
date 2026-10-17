#pragma once
#include <glm/glm.hpp>

namespace swe {

// Port of ui/camera.py: VTK / PyVista style trackball.
// Left drag = free rotation (360 deg in every direction) around the focal point, middle drag = pan,
// wheel = zoom.
class OrbitCamera
{
  public:
	OrbitCamera() = default;
	OrbitCamera(glm::vec3 position, glm::vec3 focal, glm::vec3 up = {0, 1, 0});

	void reset();
	void setHome(glm::vec3 position, glm::vec3 focal, glm::vec3 up = {0, 1, 0});

	// cursor in normalised window coordinates, origin bottom-left (like ti.ui get_cursor_pos);
	// button: 0 none, 1 left, 2 middle; blocked: the drag would start on the GUI
	void update(glm::vec2 cursor, int button, float wheel, bool blocked);

	glm::mat4 view() const;
	glm::mat4 proj(float aspect) const;        // Vulkan clip space (y down, depth 0..1)

	glm::vec3 position{}, focal{}, up{0, 1, 0};
	float     fovY = glm::radians(45.0f);      // GGUI default
	float     zNear = 0.01f, zFar = 200.0f;

  private:
	void orthogonaliseUp();

	glm::vec3 homePos_{}, homeFocal_{}, homeUp_{0, 1, 0};
	int       button_ = 0;
	glm::vec2 last_{};
};

}        // namespace swe
