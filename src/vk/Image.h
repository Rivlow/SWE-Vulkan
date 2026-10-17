#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <string>

namespace swe::vkc {

class Context;

// 2D image + dedicated memory + full view (attachments, effect fields).
class Image
{
  public:
	Image() = default;
	Image(Context const &ctx, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage,
	      vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1, std::string const &name = {});

	Image(Image &&) noexcept            = default;
	Image &operator=(Image &&) noexcept = default;

	vk::Image            image() const { return *image_; }
	vk::ImageView        view() const { return *view_; }
	vk::Format           format() const { return format_; }
	vk::Extent2D         extent() const { return extent_; }
	vk::ImageAspectFlags aspect() const { return aspect_; }
	explicit             operator bool() const { return extent_.width != 0; }

  private:
	vk::raii::Image        image_  = nullptr;
	vk::raii::DeviceMemory memory_ = nullptr;
	vk::raii::ImageView    view_   = nullptr;
	vk::Format             format_ = vk::Format::eUndefined;
	vk::Extent2D           extent_{};
	vk::ImageAspectFlags   aspect_ = vk::ImageAspectFlagBits::eColor;
};

}        // namespace swe::vkc
