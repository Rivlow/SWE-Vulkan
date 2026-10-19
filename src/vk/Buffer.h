#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <span>
#include <string>
#include <vector>

namespace swe::vkc {

class Context;

// A buffer with its own dedicated allocation (the app only needs a few dozen buffers).
class Buffer
{
  public:
	Buffer() = default;
	Buffer(Context const &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags props,
	       std::string const &name = {});

	Buffer(Buffer &&) noexcept            = default;
	Buffer &operator=(Buffer &&) noexcept = default;

	// Device-local buffer usable as storage + transfer src/dst.
	static Buffer storage(Context const &ctx, vk::DeviceSize size, std::string const &name, vk::BufferUsageFlags extra = {});
	// Host-visible, host-coherent, persistently mapped.
	static Buffer host(Context const &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, std::string const &name);
	// Host-visible transfer destination the CPU will read a lot of: host-cached when the device offers it
	// (reads from uncached memory are many times slower), else like host().
	static Buffer readback(Context const &ctx, vk::DeviceSize size, std::string const &name);

	// Synchronous upload / download through a staging buffer (case loading, dumps, tests).
	void upload(Context const &ctx, void const *data, vk::DeviceSize size, vk::DeviceSize offset = 0);
	void download(Context const &ctx, void *dst, vk::DeviceSize size, vk::DeviceSize offset = 0) const;

	template <class T>
	void upload(Context const &ctx, std::span<T const> data, vk::DeviceSize offset = 0)
	{
		upload(ctx, data.data(), data.size_bytes(), offset);
	}
	template <class T>
	std::vector<T> download(Context const &ctx, size_t count, vk::DeviceSize offset = 0) const
	{
		std::vector<T> out(count);
		download(ctx, out.data(), count * sizeof(T), offset);
		return out;
	}

	vk::Buffer               handle() const { return *buffer_; }
	vk::DeviceSize           size() const { return size_; }
	void                    *mapped() const { return mapped_; }
	vk::DescriptorBufferInfo info() const { return {.buffer = *buffer_, .offset = 0, .range = size_}; }
	explicit                 operator bool() const { return size_ != 0; }

  private:
	vk::raii::Buffer       buffer_ = nullptr;
	vk::raii::DeviceMemory memory_ = nullptr;
	vk::DeviceSize         size_   = 0;
	void                  *mapped_ = nullptr;
};

}        // namespace swe::vkc
