#include "vk/Swapchain.h"

#include "vk/Context.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <limits>

namespace swe::vkc {

Swapchain::Swapchain(Context const &ctx, GLFWwindow *window) :
    ctx_(ctx), window_(window)
{
	create();
}

void Swapchain::recreate()
{
	// Minimised window: wait until it has a non-zero size again.
	int w = 0, h = 0;
	glfwGetFramebufferSize(window_, &w, &h);
	while (w == 0 || h == 0)
	{
		glfwWaitEvents();
		glfwGetFramebufferSize(window_, &w, &h);
	}
	ctx_.device.waitIdle();
	views.clear();
	renderFinished.clear();
	create();
}

void Swapchain::create()
{
	auto caps = ctx_.physicalDevice.getSurfaceCapabilitiesKHR(*ctx_.surface);

	if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
		extent = caps.currentExtent;
	else
	{
		int w, h;
		glfwGetFramebufferSize(window_, &w, &h);
		extent = vk::Extent2D{std::clamp<uint32_t>(w, caps.minImageExtent.width, caps.maxImageExtent.width),
		                      std::clamp<uint32_t>(h, caps.minImageExtent.height, caps.maxImageExtent.height)};
	}

	minImageCount = std::max(3u, caps.minImageCount);
	if (caps.maxImageCount > 0)
		minImageCount = std::min(minImageCount, caps.maxImageCount);

	// UNORM on purpose: the palette inherited from the Python viewer (and ImGui) is authored as
	// display values, written to the framebuffer without gamma encoding.
	auto formats  = ctx_.physicalDevice.getSurfaceFormatsKHR(*ctx_.surface);
	surfaceFormat = formats.front();
	for (auto const &f : formats)
		if ((f.format == vk::Format::eB8G8R8A8Unorm || f.format == vk::Format::eR8G8B8A8Unorm) &&
		    f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
		{
			surfaceFormat = f;
			break;
		}

	auto modes   = ctx_.physicalDevice.getSurfacePresentModesKHR(*ctx_.surface);
	auto present = vk::PresentModeKHR::eFifo;        // always available, = vsync
	if (!vsync)
		for (auto m : modes)
			if (m == vk::PresentModeKHR::eMailbox || (m == vk::PresentModeKHR::eImmediate && present == vk::PresentModeKHR::eFifo))
				present = m;

	vk::SwapchainCreateInfoKHR info{.surface          = *ctx_.surface,
	                                .minImageCount    = minImageCount,
	                                .imageFormat      = surfaceFormat.format,
	                                .imageColorSpace  = surfaceFormat.colorSpace,
	                                .imageExtent      = extent,
	                                .imageArrayLayers = 1,
	                                .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
	                                .imageSharingMode = vk::SharingMode::eExclusive,
	                                .preTransform     = caps.currentTransform,
	                                .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
	                                .presentMode      = present,
	                                .clipped          = true,
	                                .oldSwapchain     = *swapchain};
	vk::raii::SwapchainKHR next(ctx_.device, info);
	swapchain = std::move(next);
	images    = swapchain.getImages();

	for (auto image : images)
	{
		views.emplace_back(ctx_.device, vk::ImageViewCreateInfo{.image            = image,
		                                                        .viewType         = vk::ImageViewType::e2D,
		                                                        .format           = surfaceFormat.format,
		                                                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
		renderFinished.emplace_back(ctx_.device, vk::SemaphoreCreateInfo{});
	}
}

void transitionImage(vk::raii::CommandBuffer const &cb, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                     vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                     vk::AccessFlags2 dstAccess, vk::ImageAspectFlags aspect)
{
	vk::ImageMemoryBarrier2 barrier{.srcStageMask        = srcStage,
	                                .srcAccessMask       = srcAccess,
	                                .dstStageMask        = dstStage,
	                                .dstAccessMask       = dstAccess,
	                                .oldLayout           = from,
	                                .newLayout           = to,
	                                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
	                                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
	                                .image               = image,
	                                .subresourceRange    = {aspect, 0, 1, 0, 1}};
	cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

}        // namespace swe::vkc
