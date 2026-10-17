#include "vk/Context.h"

#include "util/Log.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace swe::vkc {

namespace {

constexpr char const *kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                               vk::DebugUtilsMessageTypeFlagsEXT,
                                               vk::DebugUtilsMessengerCallbackDataEXT const *data, void *)
{
	if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
		log::error("[vulkan] {}", data->pMessage);
	else
		log::warn("[vulkan] {}", data->pMessage);
	return vk::False;
}

bool hasExtension(std::vector<vk::ExtensionProperties> const &props, char const *name)
{
	return std::ranges::any_of(props, [name](auto const &p) { return std::strcmp(p.extensionName, name) == 0; });
}

}        // namespace

Context::Context(ContextOptions const &opts) :
    headless(opts.window == nullptr)
{
	createInstance(opts);
	if (!headless)
	{
		VkSurfaceKHR raw = VK_NULL_HANDLE;
		if (glfwCreateWindowSurface(*instance, opts.window, nullptr, &raw) != VK_SUCCESS)
			throw std::runtime_error("glfwCreateWindowSurface failed");
		surface = vk::raii::SurfaceKHR(instance, raw);
	}
	pickPhysicalDevice(opts.forcedGpu);
	createDevice();
	commandPool = vk::raii::CommandPool(device, {.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
	                                             .queueFamilyIndex = queueFamily});
}

void Context::createInstance(ContextOptions const &opts)
{
	vk::ApplicationInfo appInfo{.pApplicationName   = "SWE-Vulkan",
	                            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
	                            .pEngineName        = "swe",
	                            .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
	                            .apiVersion         = vk::ApiVersion14};

	std::vector<char const *> layers;
	validation = false;
	if (opts.validation)
	{
		auto available = context.enumerateInstanceLayerProperties();
		if (std::ranges::any_of(available, [](auto const &l) { return std::strcmp(l.layerName, kValidationLayer) == 0; }))
		{
			layers.push_back(kValidationLayer);
			validation = true;
		}
		else
			log::warn("validation layer not found, continuing without it");
	}

	std::vector<char const *> extensions;
	if (!headless)
	{
		uint32_t     count   = 0;
		char const **glfwExt = glfwGetRequiredInstanceExtensions(&count);
		extensions.assign(glfwExt, glfwExt + count);
	}
	if (validation)
		extensions.push_back(vk::EXTDebugUtilsExtensionName);

	auto available = context.enumerateInstanceExtensionProperties();
	for (auto *e : extensions)
		if (!hasExtension(available, e))
			throw std::runtime_error(std::string("instance extension not supported: ") + e);

	std::array<vk::ValidationFeatureEnableEXT, 1> syncFeature{vk::ValidationFeatureEnableEXT::eSynchronizationValidation};
	vk::ValidationFeaturesEXT                     validationFeatures{.enabledValidationFeatureCount = 1,
	                                                                 .pEnabledValidationFeatures    = syncFeature.data()};

	vk::InstanceCreateInfo createInfo{.pNext                   = (validation && opts.syncValidation) ? &validationFeatures : nullptr,
	                                  .pApplicationInfo        = &appInfo,
	                                  .enabledLayerCount       = static_cast<uint32_t>(layers.size()),
	                                  .ppEnabledLayerNames     = layers.data(),
	                                  .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
	                                  .ppEnabledExtensionNames = extensions.data()};
	instance = vk::raii::Instance(context, createInfo);

	if (validation)
	{
		debugMessenger = instance.createDebugUtilsMessengerEXT(vk::DebugUtilsMessengerCreateInfoEXT{
		    .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		    .messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
		                   vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		    .pfnUserCallback = &debugCallback});
		log::info("validation layers on{}", opts.syncValidation ? " (+ synchronization validation)" : "");
	}
}

int Context::findQueueFamily(vk::raii::PhysicalDevice const &pd) const
{
	auto families = pd.getQueueFamilyProperties();
	for (uint32_t q = 0; q < families.size(); ++q)
	{
		auto flags = families[q].queueFlags;
		if ((flags & vk::QueueFlagBits::eGraphics) && (flags & vk::QueueFlagBits::eCompute) &&
		    (headless || pd.getSurfaceSupportKHR(q, *surface)))
			return static_cast<int>(q);
	}
	return -1;
}

bool Context::isSuitable(vk::raii::PhysicalDevice const &pd) const
{
	if (pd.getProperties().apiVersion < vk::ApiVersion13)
		return false;
	auto f = pd.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features>();
	if (!f.get<vk::PhysicalDeviceVulkan12Features>().timelineSemaphore ||
	    !f.get<vk::PhysicalDeviceVulkan12Features>().scalarBlockLayout ||
	    !f.get<vk::PhysicalDeviceVulkan13Features>().synchronization2 ||
	    !f.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering)
		return false;
	if (!headless)
	{
		if (!hasExtension(pd.enumerateDeviceExtensionProperties(), vk::KHRSwapchainExtensionName))
			return false;
		// the renderer reads the solver's storage buffers from vertex/fragment shaders and blends water
		// with a second fragment output
		auto core = pd.getFeatures();
		if (!core.dualSrcBlend || !core.vertexPipelineStoresAndAtomics || !core.fragmentStoresAndAtomics)
			return false;
	}
	return findQueueFamily(pd) >= 0;
}

void Context::pickPhysicalDevice(int forced)
{
	auto devices = instance.enumeratePhysicalDevices();
	if (devices.empty())
		throw std::runtime_error("no Vulkan device found");

	int best = -1, bestScore = -1;
	for (int i = 0; i < static_cast<int>(devices.size()); ++i)
	{
		auto props = devices[i].getProperties();
		bool ok    = isSuitable(devices[i]);
		log::info("GPU {}: {} ({}){}", i, props.deviceName.data(), vk::to_string(props.deviceType), ok ? "" : " [unsuitable]");
		if (!ok)
			continue;
		int score = props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu   ? 1000 :
		            props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu ? 100 :
		                                                                         10;
		if (i == forced)
			score = 1 << 20;
		if (score > bestScore)
		{
			best      = i;
			bestScore = score;
		}
	}
	if (forced >= 0 && best != forced)
		log::warn("--gpu {} is not usable, falling back to automatic selection", forced);
	if (best < 0)
		throw std::runtime_error("no suitable GPU (needs Vulkan 1.3, synchronization2, dynamic rendering, timeline semaphores)");

	physicalDevice = devices[best];
	properties     = physicalDevice.getProperties();
	queueFamily    = static_cast<uint32_t>(findQueueFamily(physicalDevice));
	log::info("using GPU {}: {}", best, properties.deviceName.data());
}

void Context::createDevice()
{
	auto supported = physicalDevice.getFeatures();

	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan12Features,
	                   vk::PhysicalDeviceVulkan13Features>
	    features{
	        {.features = {.dualSrcBlend                   = supported.dualSrcBlend,        // water transparency
	                      .fillModeNonSolid               = supported.fillModeNonSolid,    // wireframe overlay
	                      .wideLines                      = supported.wideLines,
	                      .samplerAnisotropy              = supported.samplerAnisotropy,
	                      .vertexPipelineStoresAndAtomics = supported.vertexPipelineStoresAndAtomics,
	                      .fragmentStoresAndAtomics       = supported.fragmentStoresAndAtomics}},
	        {.shaderDrawParameters = true},        // SV_VertexID is lowered to VertexIndex - BaseVertex
	        {.scalarBlockLayout = true, .timelineSemaphore = true},
	        {.synchronization2 = true, .dynamicRendering = true},
	    };

	std::vector<char const *> extensions;
	if (!headless)
		extensions.push_back(vk::KHRSwapchainExtensionName);

	float                     priority = 1.0f;
	vk::DeviceQueueCreateInfo queueInfo{.queueFamilyIndex = queueFamily, .queueCount = 1, .pQueuePriorities = &priority};
	device = vk::raii::Device(physicalDevice, vk::DeviceCreateInfo{.pNext                   = &features.get<vk::PhysicalDeviceFeatures2>(),
	                                                               .queueCreateInfoCount    = 1,
	                                                               .pQueueCreateInfos       = &queueInfo,
	                                                               .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
	                                                               .ppEnabledExtensionNames = extensions.data()});
	queue  = vk::raii::Queue(device, queueFamily, 0);
}

uint32_t Context::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags props) const
{
	auto mem = physicalDevice.getMemoryProperties();
	for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
		if ((typeFilter & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
			return i;
	throw std::runtime_error("no suitable memory type");
}

void Context::oneShot(std::function<void(vk::raii::CommandBuffer &)> const &fn) const
{
	auto  cbs = vk::raii::CommandBuffers(device, {.commandPool = *commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
	auto &cb  = cbs.front();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	fn(cb);
	cb.end();
	vk::raii::Fence             fence(device, vk::FenceCreateInfo{});
	vk::CommandBufferSubmitInfo cbInfo{.commandBuffer = *cb};
	queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &cbInfo}, *fence);
	if (device.waitForFences(*fence, vk::True, UINT64_MAX) != vk::Result::eSuccess)
		throw std::runtime_error("oneShot: fence wait failed");
}

}        // namespace swe::vkc
