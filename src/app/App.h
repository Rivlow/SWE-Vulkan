#pragma once
#include "cases/Cases.h"
#include "render/Camera.h"
#include "render/Renderer.h"
#include "solver/Solver.h"
#include "util/Cli.h"
#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Swapchain.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <optional>

struct GLFWwindow;

namespace swe {

// Interactive application: window, frame loop, camera, GUI. Owns the solver and the renderer.
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
	static_assert(kFramesInFlight == Renderer::kFrames);

	struct Frame
	{
		vk::raii::CommandBuffer cmd            = nullptr;
		vk::raii::Fence         inFlight       = nullptr;
		vk::raii::Semaphore     imageAvailable = nullptr;
	};

	void initWindow();
	void initImGui();
	void shutdownImGui();
	void loadCase(std::string const &name);
	void handleInput();
	void drawGui();
	void drawControls();
	void resetSimulation();
	void drawFrame();
	void recordFrame(Frame &frame, uint32_t imageIndex);
	void recordCapture(vk::raii::CommandBuffer const &cb, vk::Image image);
	void saveCapture();
	void onSwapchainRecreated();

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

	Options                            opts_;
	GLFWwindow                        *window_ = nullptr;
	std::unique_ptr<vkc::Context>      ctx_;
	std::unique_ptr<vkc::Swapchain>    swapchain_;
	std::unique_ptr<Solver>            solver_;
	std::unique_ptr<Renderer>          renderer_;
	std::array<Frame, kFramesInFlight> frames_;
	uint32_t                           frameIndex_         = 0;
	uint64_t                           frameCount_         = 0;
	bool                               framebufferResized_ = false;
	bool                               imguiReady_         = false;
	vk::Format                         colorFormat_        = vk::Format::eUndefined;

	// Simulation state shown in the GUI (scalars lag one frame behind: read from the slot just completed)
	SolverScalars scalars_{};
	uint32_t      prevSteps_     = 0;
	double        stepsPerSec_   = 0.0;
	bool          paused_        = false;
	int           stepsPerFrame_ = 30;        // main.py: steps_per_render

	// View
	OrbitCamera    camera_;
	RenderSettings render_;
	float          zScale0_ = 2.0f;

	// Screenshot (F12, --capture): swapchain image copied to a host buffer, written after the frame's fence
	std::optional<std::filesystem::path> captureRequest_;
	std::optional<std::filesystem::path> capturePending_;
	vkc::Buffer                          captureBuffer_;
	vk::Extent2D                         captureExtent_{};

	// GUI state
	std::vector<CaseInfo> caseList_;
	bool                  showDemo_ = false;
};

}        // namespace swe
