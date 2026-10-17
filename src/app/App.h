#pragma once
#include "app/MapPanel.h"
#include "cases/Cases.h"
#include "geo/Carve.h"
#include "geo/LocalLibrary.h"
#include "geo/TerrainBuilder.h"
#include "geo/Tiles.h"
#include "render/Camera.h"
#include "render/Effects.h"
#include "render/Renderer.h"
#include "solver/Solver.h"
#include "util/Cli.h"
#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Swapchain.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct GLFWwindow;

namespace swe {

// Interactive application: window, frame loop, camera, GUI. Owns the solver, the effects and the renderer.
// Frame loop in App.cpp, GUI panels in Gui.cpp.
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
	static constexpr uint32_t kTimestamps = 4;        // per frame: before steps, after steps, after effects, end

	struct Frame
	{
		vk::raii::CommandBuffer cmd            = nullptr;
		vk::raii::CommandBuffer begin          = nullptr;        // query reset + first timestamp (static)
		vk::raii::Fence         inFlight       = nullptr;
		vk::raii::Semaphore     imageAvailable = nullptr;
		vkc::Buffer             capture;                        // screenshot copy of this frame
		std::optional<std::filesystem::path> capturePath;       // pending screenshot of this frame
		vk::Extent2D            captureExtent{};
	};

	struct Probe
	{
		uint32_t i = 0, j = 0;
	};

	// time series of the plots (t on the x axis), decimated when they grow too long
	struct History
	{
		std::vector<double>                          t, volume, wetArea, dt;
		std::array<std::vector<double>, MAX_PROBES>  eta, depth, speed;
		void clear();
		void push(SolverScalars const &s, double cellArea, size_t probes);
		void decimate();
	};

	struct GpuTimes
	{
		double solver = 0, effects = 0, render = 0;        // ms, smoothed
	};

	// App.cpp - frame loop
	void initWindow();
	void initImGui();
	void shutdownImGui();
	void createTimestamps();
	void loadCase(std::string const &name);
	void loadImport(ImportOutcome &&outcome);
	void applyBedEdit(BedEdit const &edit);
	void saveCurrentCase();
	void resetSimulation();
	void topView();
	void addZoneFromSelection(ZoneKind kind, double value, std::string const &label);
	void removeZone(size_t index);
	void updateZoneValue(size_t index, double value);
	void handleInput();
	void drawFrame();
	void recordFrame(Frame &frame, uint32_t imageIndex);
	void recordCapture(Frame &frame, vk::raii::CommandBuffer const &cb, vk::Image image);
	void saveCapture(Frame &frame);
	void readTimestamps(uint32_t slot);
	void onSwapchainRecreated();
	void loadSettings();
	void saveSettings() const;

	// Gui.cpp - panels
	void drawGui();
	void setupDockLayout(unsigned int dockspace);
	void drawMenuBar();
	void drawSimulationPanel();
	void drawSchemePanel();
	void drawDisplayPanel();
	void drawProbesPanel();
	void drawEditPanel();
	void drawPlotsPanel();
	void drawPerfPanel();
	void drawLegend();
	void drawProbeMarkers();
	void drawHelp();
	std::optional<Probe> pickCell(glm::vec2 mousePixels) const;
	void                 updateProbeList();
	void                 autoColorRange();

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

	Options                            opts_;
	GLFWwindow                        *window_ = nullptr;
	std::unique_ptr<vkc::Context>      ctx_;
	std::unique_ptr<vkc::Swapchain>    swapchain_;
	std::unique_ptr<Solver>            solver_;
	std::unique_ptr<Effects>           effects_;
	std::unique_ptr<Renderer>          renderer_;
	std::unique_ptr<geo::LocalLibrary>   library_;
	std::unique_ptr<geo::TerrainBuilder> builder_;
	std::unique_ptr<MapPanel>            map_;
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
	double        volume0_       = -1.0;      // initial volume (mass balance)
	History       history_;

	// Georeferenced case (imports): grid origin and the orthophoto bytes (re-applied after bed edits)
	std::optional<geo::GridGeometry> caseGeo_;
	geo::Image                       orthoImage_;

	// View
	OrbitCamera    camera_;
	RenderSettings render_;
	float          zScale0_  = 2.0f;
	bool           autoRange_ = true;
	glm::vec4      viewRect_{0.0f};           // central dock area (pixels)

	// Probes
	std::vector<Probe> probes_;

	// Cell selection in the 3D view (edit tool)
	bool                 selectTool_ = false;
	bool                 selecting_  = false;
	std::optional<Probe> selAnchor_, selCurrent_;
	// edit panel state
	int   waterMode_   = 0;          // 0 depth, 1 level, 2 add
	float waterValue_  = 1.0f;
	bool  waterInitial_ = true;
	int   zoneKind_    = 0;          // 0 source, 1 sink, 2 level
	float zoneValue_   = 10.0f;
	char  zoneLabel_[48] = "";

	// GPU timing
	vk::raii::QueryPool queries_         = nullptr;
	double              timestampPeriod_ = 0.0;        // ns per tick, 0 = unsupported
	GpuTimes            gpu_;

	// Screenshots (F12, --capture) and recording (PNG sequence written by worker threads)
	std::optional<std::filesystem::path> captureRequest_;
	bool                                 captureDone_ = false;
	bool                                 recording_   = false;
	int                                  recordEvery_ = 1;
	uint32_t                             recordIndex_ = 0;
	std::filesystem::path                recordDir_;
	std::vector<std::future<bool>>       writers_;

	// GUI state
	std::vector<CaseInfo> caseList_;
	bool showSimulation_ = true, showScheme_ = true, showDisplay_ = true, showProbes_ = true;
	bool showPlots_ = true, showPerf_ = false, showHelp_ = false, showDemo_ = false, showMap_ = true, showEdit_ = true;
	bool hideGui_ = false;        // F11: 3D view only
};

}        // namespace swe
