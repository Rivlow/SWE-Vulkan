#include "vk/Texture.h"

#include "vk/Buffer.h"
#include "vk/Context.h"

#include <algorithm>
#include <cmath>

namespace swe::vkc {

Texture::Texture(Context const &ctx, uint32_t width, uint32_t height, uint8_t const *rgba, bool mipmaps, std::string const &name) :
    width_(width), height_(height)
{
	mipLevels_ = mipmaps ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1 : 1;
	auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
	image_     = vk::raii::Image(ctx.device, vk::ImageCreateInfo{.imageType     = vk::ImageType::e2D,
	                                                             .format        = vk::Format::eR8G8B8A8Unorm,
	                                                             .extent        = {width, height, 1},
	                                                             .mipLevels     = mipLevels_,
	                                                             .arrayLayers   = 1,
	                                                             .samples       = vk::SampleCountFlagBits::e1,
	                                                             .tiling        = vk::ImageTiling::eOptimal,
	                                                             .usage         = usage,
	                                                             .sharingMode   = vk::SharingMode::eExclusive,
	                                                             .initialLayout = vk::ImageLayout::eUndefined});
	auto reqs  = image_.getMemoryRequirements();
	memory_    = vk::raii::DeviceMemory(ctx.device, {.allocationSize  = reqs.size,
	                                                 .memoryTypeIndex = ctx.findMemoryType(reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
	image_.bindMemory(*memory_, 0);

	Buffer staging = Buffer::host(ctx, vk::DeviceSize(width) * height * 4, vk::BufferUsageFlagBits::eTransferSrc, "texture-staging");
	std::memcpy(staging.mapped(), rgba, staging.size());

	ctx.oneShot([&](vk::raii::CommandBuffer &cb) {
		auto range = [&](uint32_t base, uint32_t count) {
			return vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, base, count, 0, 1};
		};
		auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::PipelineStageFlags2 ss, vk::AccessFlags2 sa,
		                   vk::PipelineStageFlags2 ds, vk::AccessFlags2 da, uint32_t base, uint32_t count) {
			vk::ImageMemoryBarrier2 b{.srcStageMask        = ss,
			                          .srcAccessMask       = sa,
			                          .dstStageMask        = ds,
			                          .dstAccessMask       = da,
			                          .oldLayout           = from,
			                          .newLayout           = to,
			                          .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
			                          .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
			                          .image               = *image_,
			                          .subresourceRange    = range(base, count)};
			cb.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
		};
		barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eTopOfPipe, {},
		        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite, 0, mipLevels_);
		cb.copyBufferToImage(staging.handle(), *image_, vk::ImageLayout::eTransferDstOptimal,
		                     vk::BufferImageCopy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
		                                         .imageExtent      = {width, height, 1}});
		// mip chain by successive blits
		int32_t w = int32_t(width), h = int32_t(height);
		for (uint32_t level = 1; level < mipLevels_; ++level)
		{
			barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits2::eTransfer,
			        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
			        level - 1, 1);
			int32_t nw = std::max(w / 2, 1), nh = std::max(h / 2, 1);
			vk::ImageBlit blit{.srcSubresource = {vk::ImageAspectFlagBits::eColor, level - 1, 0, 1},
			                   .srcOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{w, h, 1}}},
			                   .dstSubresource = {vk::ImageAspectFlagBits::eColor, level, 0, 1},
			                   .dstOffsets     = {{vk::Offset3D{0, 0, 0}, vk::Offset3D{nw, nh, 1}}}};
			cb.blitImage(*image_, vk::ImageLayout::eTransferSrcOptimal, *image_, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
			barrier(vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
			        vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
			        level - 1, 1);
			w = nw;
			h = nh;
		}
		barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eTransfer,
		        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
		        mipLevels_ - 1, 1);
	});

	view_    = vk::raii::ImageView(ctx.device, vk::ImageViewCreateInfo{.image            = *image_,
	                                                                   .viewType         = vk::ImageViewType::e2D,
	                                                                   .format           = vk::Format::eR8G8B8A8Unorm,
	                                                                   .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, mipLevels_, 0, 1}});
	sampler_ = vk::raii::Sampler(ctx.device, vk::SamplerCreateInfo{.magFilter        = vk::Filter::eLinear,
	                                                               .minFilter        = vk::Filter::eLinear,
	                                                               .mipmapMode       = vk::SamplerMipmapMode::eLinear,
	                                                               .addressModeU     = vk::SamplerAddressMode::eClampToEdge,
	                                                               .addressModeV     = vk::SamplerAddressMode::eClampToEdge,
	                                                               .addressModeW     = vk::SamplerAddressMode::eClampToEdge,
	                                                               .anisotropyEnable = mipmaps,
	                                                               .maxAnisotropy    = mipmaps ? 8.0f : 1.0f,
	                                                               .maxLod           = float(mipLevels_)});
	if (!name.empty())
		ctx.setName(image_, name);
}

Texture Texture::white(Context const &ctx)
{
	uint8_t px[4] = {255, 255, 255, 255};
	return Texture(ctx, 1, 1, px, false, "white");
}

}        // namespace swe::vkc
