#pragma once
#include <glm/glm.hpp>

namespace swe {

// Port of ui/camera.py: VTK / PyVista style trackball.
// Left drag = free rotation (360 deg in every direction) around the focal point, middle drag = pan,
// wheel = zoom.
class OrbitCamera
{
  public:
	struct Pose
	{
		glm::vec3 position{}, focal{}, up{0, 1, 0};
	};

	OrbitCamera() = default;
	OrbitCamera(glm::vec3 position, glm::vec3 focal, glm::vec3 up = {0, 1, 0});

	void reset();
	void setHome(glm::vec3 position, glm::vec3 focal, glm::vec3 up = {0, 1, 0});
	Pose home() const { return {homePos_, homeFocal_, homeUp_}; }
	Pose pose() const { return {position, focal, up}; }
	void setPose(Pose const &p);
	// Straight down onto `focal`, north (display -Z) at the top of the screen, `extent` display units visible
	void topView(glm::vec3 focalPoint, float extent);
	static Pose topPose(glm::vec3 focalPoint, float extent, float fovY);

	// Smooth move to `target` over `seconds` (eased); tick(dt) advances it, a drag or stop() ends it
	void animateTo(Pose const &target, float seconds);
	void tick(float dt);
	void stop() { animating_ = false; }
	bool animating() const { return animating_; }

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

	// animation
	bool  animating_ = false;
	Pose  from_{}, to_{};
	float animT_ = 0.0f, animDuration_ = 1.0f;
};

}        // namespace swe
