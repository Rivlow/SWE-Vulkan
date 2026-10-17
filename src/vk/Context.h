#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <functional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace swe::vkc {

struct ContextOptions
{
	bool        validation     = true;
	bool        syncValidation = false;
	int         forcedGpu      = -1;         // index in enumeratePhysicalDevices(), -1 = auto (prefer discrete)
	GLFWwindow *window         = nullptr;    // nullptr = headless (no surface, no swapchain)
};

// Instance, device and the single graphics+compute(+present) queue shared by the solver and the renderer.
class Context
{
  public:
	explicit Context(ContextOptions const &opts);

	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::raii::Device                 device         = nullptr;
	vk::raii::Queue                  queue          = nullptr;
	uint32_t                         queueFamily    = ~0u;
	vk::raii::CommandPool            commandPool    = nullptr;        // resettable command buffers

	vk::PhysicalDeviceProperties properties;
	bool                         headless   = true;
	bool                         validation = false;

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags props) const;

	// Records `fn` into a temporary command buffer, submits it and waits for completion.
	void oneShot(std::function<void(vk::raii::CommandBuffer &)> const &fn) const;

	// Attach a debug name to any vk::raii handle (no-op without validation).
	template <class Handle>
	void setName(Handle const &handle, std::string const &name) const
	{
		if (!validation)
			return;
		using CType = typename Handle::CType;
		device.setDebugUtilsObjectNameEXT(vk::DebugUtilsObjectNameInfoEXT{
		    .objectType   = Handle::objectType,
		    .objectHandle = reinterpret_cast<uint64_t>(static_cast<CType>(*handle)),
		    .pObjectName  = name.c_str()});
	}

  private:
	void createInstance(ContextOptions const &opts);
	void pickPhysicalDevice(int forced);
	void createDevice();
	bool isSuitable(vk::raii::PhysicalDevice const &pd) const;
	int  findQueueFamily(vk::raii::PhysicalDevice const &pd) const;
};

}        // namespace swe::vkc
