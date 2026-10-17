#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <string>

namespace swe::vkc {

class Context;

// Sampled 2D RGBA8 texture uploaded from host pixels (map tiles, orthophoto), with mipmaps.
class Texture
{
  public:
	Texture() = default;
	Texture(Context const &ctx, uint32_t width, uint32_t height, uint8_t const *rgba, bool mipmaps, std::string const &name = {});

	Texture(Texture &&) noexcept            = default;
	Texture &operator=(Texture &&) noexcept = default;

	vk::ImageView view() const { return *view_; }
	vk::Sampler   sampler() const { return *sampler_; }
	uint32_t      width() const { return width_; }
	uint32_t      height() const { return height_; }
	explicit      operator bool() const { return width_ != 0; }

	// 1x1 white texture (placeholder when nothing is loaded)
	static Texture white(Context const &ctx);

  private:
	vk::raii::Image        image_   = nullptr;
	vk::raii::DeviceMemory memory_  = nullptr;
	vk::raii::ImageView    view_    = nullptr;
	vk::raii::Sampler      sampler_ = nullptr;
	uint32_t               width_ = 0, height_ = 0, mipLevels_ = 1;
};

}        // namespace swe::vkc
