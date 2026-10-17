#include "vk/Image.h"

#include "vk/Context.h"

namespace swe::vkc {

Image::Image(Context const &ctx, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage,
             vk::SampleCountFlagBits samples, std::string const &name) :
    format_(format), extent_(extent)
{
	bool depth = format == vk::Format::eD32Sfloat || format == vk::Format::eD24UnormS8Uint || format == vk::Format::eD16Unorm;
	aspect_    = depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;

	image_    = vk::raii::Image(ctx.device, vk::ImageCreateInfo{.imageType     = vk::ImageType::e2D,
	                                                            .format        = format,
	                                                            .extent        = {extent.width, extent.height, 1},
	                                                            .mipLevels     = 1,
	                                                            .arrayLayers   = 1,
	                                                            .samples       = samples,
	                                                            .tiling        = vk::ImageTiling::eOptimal,
	                                                            .usage         = usage,
	                                                            .sharingMode   = vk::SharingMode::eExclusive,
	                                                            .initialLayout = vk::ImageLayout::eUndefined});
	auto reqs = image_.getMemoryRequirements();
	memory_   = vk::raii::DeviceMemory(ctx.device, {.allocationSize  = reqs.size,
	                                                .memoryTypeIndex = ctx.findMemoryType(reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
	image_.bindMemory(*memory_, 0);
	view_ = vk::raii::ImageView(ctx.device, vk::ImageViewCreateInfo{.image            = *image_,
	                                                                .viewType         = vk::ImageViewType::e2D,
	                                                                .format           = format,
	                                                                .subresourceRange = {aspect_, 0, 1, 0, 1}});
	if (!name.empty())
	{
		ctx.setName(image_, name);
		ctx.setName(view_, name);
	}
}

}        // namespace swe::vkc
