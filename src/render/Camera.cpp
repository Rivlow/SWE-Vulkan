#include "render/Camera.h"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace swe {

namespace {

// rotation of v around the unit axis - Rodrigues' formula
glm::vec3 rotate(glm::vec3 v, glm::vec3 axis, float angle)
{
	float c = std::cos(angle), s = std::sin(angle);
	return v * c + glm::cross(axis, v) * s + axis * glm::dot(axis, v) * (1.0f - c);
}

}        // namespace

OrbitCamera::OrbitCamera(glm::vec3 homePosition, glm::vec3 homeFocal, glm::vec3 homeUp)
{
	setHome(homePosition, homeFocal, homeUp);
}

void OrbitCamera::setHome(glm::vec3 homePosition, glm::vec3 homeFocal, glm::vec3 homeUp)
{
	homePos_   = homePosition;
	homeFocal_ = homeFocal;
	homeUp_    = homeUp;
	reset();
}

void OrbitCamera::reset()
{
	position = homePos_;
	focal    = homeFocal_;
	up       = homeUp_;
	orthogonaliseUp();
	button_ = 0;
}

void OrbitCamera::topView(glm::vec3 focalPoint, float extent)
{
	float dist = 0.5f * extent / std::tan(0.5f * fovY) * 1.08f;
	focal      = focalPoint;
	position   = focalPoint + glm::vec3(0.0f, dist, 0.0f);
	up         = glm::vec3(0.0f, 0.0f, -1.0f);
	orthogonaliseUp();
	button_ = 0;
}

void OrbitCamera::orthogonaliseUp()
{
	glm::vec3 forward = glm::normalize(focal - position);
	glm::vec3 right   = glm::normalize(glm::cross(forward, up));
	up                = glm::cross(right, forward);
}

void OrbitCamera::update(glm::vec2 cursor, int button, float wheel, bool blocked)
{
	if (wheel != 0.0f && !blocked)        // zoom: wheel forward = closer
		position = focal + (position - focal) * std::exp(-0.15f * wheel);

	if (button == 0)
	{
		button_ = 0;
		return;
	}
	if (button_ == 0)        // start of a drag (ignored when it starts on the GUI)
	{
		if (blocked)
			return;
		button_ = button;
		last_   = cursor;
		return;
	}
	glm::vec2 d = cursor - last_;
	last_       = cursor;

	glm::vec3 offset  = position - focal;
	float     dist    = glm::length(offset);
	glm::vec3 forward = -offset / dist;
	glm::vec3 right   = glm::normalize(glm::cross(forward, up));

	if (button_ == 1)        // rotation: azimuth around "up", elevation around "right"
	{
		offset   = rotate(offset, up, -3.0f * d.x);
		up       = rotate(up, right, 3.0f * d.y);        // up turns too -> no stop at the poles
		offset   = rotate(offset, right, 3.0f * d.y);
		position = focal + offset;
	}
	else if (button_ == 2)        // pan the camera AND the focal point
	{
		glm::vec3 shift = (-d.x * right - d.y * up) * dist;
		position += shift;
		focal += shift;
	}
	orthogonaliseUp();
}

glm::mat4 OrbitCamera::view() const
{
	return glm::lookAt(position, focal, up);
}

glm::mat4 OrbitCamera::proj(float aspect) const
{
	glm::mat4 p = glm::perspective(fovY, aspect, zNear, zFar);
	p[1][1] *= -1.0f;        // Vulkan: y down in clip space
	return p;
}

}        // namespace swe
