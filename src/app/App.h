#pragma once
#include "util/Cli.h"
#include "vk/Context.h"
#include "vk/Swapchain.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <memory>

struct GLFWwindow;

namespace swe {

// Interactive application: window, frame loop, GUI. The solver and renderer plug in from M1/M2.
class App
{
  public:
	explicit App(Options const &opts);
	~App();

	App(App const &)            = delete;
	App &operator=(App const &) = delete;

	void run();

  private:
	static constexpr uint32_t kFramesInFlight = 2;

	struct Frame
	{
		vk::raii::CommandBuffer cmd            = nullptr;
		vk::raii::Fence         inFlight       = nullptr;
		vk::raii::Semaphore     imageAvailable = nullptr;
	};

	void initWindow();
	void initImGui();
	void shutdownImGui();
	void drawGui();
	void drawFrame();
	void recordFrame(Frame &frame, uint32_t imageIndex);
	void onSwapchainRecreated();

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

	Options                          opts_;
	GLFWwindow                      *window_ = nullptr;
	std::unique_ptr<vkc::Context>    ctx_;
	std::unique_ptr<vkc::Swapchain>  swapchain_;
	std::array<Frame, kFramesInFlight> frames_;
	uint32_t                         frameIndex_         = 0;
	bool                             framebufferResized_ = false;
	bool                             imguiReady_         = false;
	vk::Format                       colorFormat_        = vk::Format::eUndefined;

	// GUI state
	bool showDemo_ = false;
};

}        // namespace swe
