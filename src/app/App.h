#pragma once
#include "solver/Solver.h"
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
	static constexpr uint32_t kFramesInFlight = Solver::kReadbackSlots;

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
	void drawControls();
	void resetSimulation();
	void drawFrame();
	void recordFrame(Frame &frame, uint32_t imageIndex);
	void onSwapchainRecreated();

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

	Options                          opts_;
	GLFWwindow                      *window_ = nullptr;
	std::unique_ptr<vkc::Context>    ctx_;
	std::unique_ptr<vkc::Swapchain>  swapchain_;
	std::unique_ptr<Solver>          solver_;
	std::array<Frame, kFramesInFlight> frames_;
	uint32_t                         frameIndex_         = 0;
	bool                             framebufferResized_ = false;
	bool                             imguiReady_         = false;
	vk::Format                       colorFormat_        = vk::Format::eUndefined;

	// Simulation state shown in the GUI (scalars lag one frame behind: read from the slot just completed)
	SolverScalars scalars_{};
	uint32_t      prevSteps_     = 0;
	double        stepsPerSec_   = 0.0;
	bool          paused_        = false;
	int           stepsPerFrame_ = 30;        // main.py: steps_per_render

	// GUI state
	bool showDemo_ = false;
};

}        // namespace swe
