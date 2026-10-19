#include "vk/Buffer.h"

#include "vk/Context.h"

#include <cstring>
#include <stdexcept>

namespace swe::vkc {

Buffer::Buffer(Context const &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props,
               std::string const &name) :
    size_(size)
{
	buffer_   = vk::raii::Buffer(ctx.device, {.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive});
	auto reqs = buffer_.getMemoryRequirements();
	memory_   = vk::raii::DeviceMemory(ctx.device, {.allocationSize  = reqs.size,
	                                                .memoryTypeIndex = ctx.findMemoryType(reqs.memoryTypeBits, props)});
	buffer_.bindMemory(*memory_, 0);
	if (props & vk::MemoryPropertyFlagBits::eHostVisible)
		mapped_ = memory_.mapMemory(0, size);
	if (!name.empty())
		ctx.setName(buffer_, name);
}

Buffer Buffer::storage(Context const &ctx, vk::DeviceSize size, std::string const &name, vk::BufferUsageFlags extra)
{
	return Buffer(ctx, size,
	              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
	                  vk::BufferUsageFlagBits::eTransferDst | extra,
	              vk::MemoryPropertyFlagBits::eDeviceLocal, name);
}

Buffer Buffer::host(Context const &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, std::string const &name)
{
	return Buffer(ctx, size, usage, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, name);
}

Buffer Buffer::readback(Context const &ctx, vk::DeviceSize size, std::string const &name)
{
	auto const usage = vk::BufferUsageFlagBits::eTransferDst;
	try
	{
		return Buffer(ctx, size, usage,
		              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
		              name);
	}
	catch (std::runtime_error const &)
	{
		return host(ctx, size, usage, name);
	}
}

void Buffer::upload(Context const &ctx, void const *data, vk::DeviceSize size, vk::DeviceSize offset)
{
	if (size == 0)
		return;
	if (mapped_)
	{
		std::memcpy(static_cast<char *>(mapped_) + offset, data, size);
		return;
	}
	Buffer staging = host(ctx, size, vk::BufferUsageFlagBits::eTransferSrc, "staging-upload");
	std::memcpy(staging.mapped(), data, size);
	ctx.oneShot([&](vk::raii::CommandBuffer &cb) {
		cb.copyBuffer(staging.handle(), *buffer_, vk::BufferCopy{.srcOffset = 0, .dstOffset = offset, .size = size});
	});
}

void Buffer::download(Context const &ctx, void *dst, vk::DeviceSize size, vk::DeviceSize offset) const
{
	if (size == 0)
		return;
	if (mapped_)
	{
		std::memcpy(dst, static_cast<char const *>(mapped_) + offset, size);
		return;
	}
	Buffer staging = host(ctx, size, vk::BufferUsageFlagBits::eTransferDst, "staging-download");
	ctx.oneShot([&](vk::raii::CommandBuffer &cb) {
		// Make every earlier write (shaders, transfers) visible to the copy.
		vk::MemoryBarrier2 barrier{.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
		                           .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
		                           .dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
		                           .dstAccessMask = vk::AccessFlagBits2::eTransferRead};
		cb.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
		cb.copyBuffer(*buffer_, staging.handle(), vk::BufferCopy{.srcOffset = offset, .dstOffset = 0, .size = size});
	});
	std::memcpy(dst, staging.mapped(), size);
}

}        // namespace swe::vkc
