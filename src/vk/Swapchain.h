#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <vector>

struct GLFWwindow;

namespace swe::vkc {

class Context;

// Swapchain + image views. One "render finished" semaphore per swapchain image
// (a present may still be reading it when the frame-in-flight slot comes around).
class Swapchain
{
  public:
	Swapchain(Context const &ctx, GLFWwindow *window);

	// Recreates the swapchain for the current framebuffer size. Waits for the device to be idle.
	void recreate();

	vk::raii::SwapchainKHR           swapchain = nullptr;
	std::vector<vk::Image>           images;
	std::vector<vk::raii::ImageView> views;
	std::vector<vk::raii::Semaphore> renderFinished;
	vk::SurfaceFormatKHR             surfaceFormat;
	vk::Extent2D                     extent;
	uint32_t                         minImageCount = 3;
	bool                             vsync         = true;

	uint32_t imageCount() const { return static_cast<uint32_t>(images.size()); }

  private:
	void create();

	Context const &ctx_;
	GLFWwindow    *window_;
};

// Image layout transition on the whole colour subresource (synchronization2).
void transitionImage(vk::raii::CommandBuffer const &cb, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                     vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                     vk::AccessFlags2 dstAccess, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor);

}        // namespace swe::vkc
