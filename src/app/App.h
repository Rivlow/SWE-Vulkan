#pragma once
#include "app/Cli.h"
#include "app/History.h"
#include "app/MapPanel.h"
#include "cases/Cases.h"
#include "geo/Carve.h"
#include "geo/LocalLibrary.h"
#include "geo/TerrainBuilder.h"
#include "geo/Tiles.h"
#include "render/Camera.h"
#include "render/Effects.h"
#include "render/Renderer.h"
#include "sim/Session.h"
#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Swapchain.h"

#include <imgui.h>
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

// Interactive application: window, frame loop, camera, GUI. Owns the simulation session, the effects and the
// renderer; every edit of the case goes through the Session. Frame loop in App.cpp, GUI in Gui.cpp.
//
// The GUI is a four-stage workflow, each stage with its own side panel:
//   Map        choose an area on the 2D map and import it (or open a saved case)
//   Terrain    the imported relief in 3D (animated transition from the map)
//   Conditions initial / boundary conditions on the mesh, paused at t = 0
//   Simulation run, balance, display, plots
class App
{
  public:
	explicit App(Options const &opts);
	~App();

	App(App const &)            = delete;
	App &operator=(App const &) = delete;

	void run();

	enum class Stage
	{
		Map,
		Terrain,
		Conditions,
		Simulation,
	};

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

	struct GpuTimes
	{
		double solver = 0, effects = 0, render = 0;        // ms, smoothed
	};

	// Map -> Terrain transition: the map fades out over the top view of the relief, then the camera swings to
	// the 3D view while the relief extrudes (z_scale 0 -> case value)
	struct Transition
	{
		bool  active = false;
		float t      = 0.0f;        // seconds since the start
	};
	static constexpr float kFadeSeconds = 0.7f, kSwingStart = 0.45f, kSwingSeconds = 2.2f;

	// App.cpp - frame loop
	void initWindow();
	void initImGui();
	void shutdownImGui();
	void createTimestamps();
	void loadCase(std::string const &name);
	void onCaseLoaded();
	void loadImport(ImportOutcome &&outcome);
	void applyBedEdit(BedEdit const &edit);
	void processMapRequests();
	void saveCurrentCase();
	void resetSimulation();
	void topView();
	void animateCamera(OrbitCamera::Pose const &target, float seconds = 1.0f);
	OrbitCamera::Pose topPose() const;
	CellRect selection() const;
	void applyScriptedEdits();
	void handleInput();
	void drawFrame();
	void recordFrame(Frame &frame, uint32_t imageIndex);
	void recordCapture(Frame &frame, vk::raii::CommandBuffer const &cb, vk::Image image);
	void saveCapture(Frame &frame);
	void readTimestamps(uint32_t slot);
	void onSwapchainRecreated();
	void loadSettings();
	void saveSettings() const;
	// stages
	void setStage(Stage s);
	void openCase(std::string const &name);        // from the Map stage: load and show the terrain
	void startTransition();
	void tickTransition(float dt);

	// Gui.cpp - stage bar, side panel, plots band, overlays
	void drawGui();
	void drawMenuBar();
	void drawStageBar(float y, float height);
	void drawSidePanel(ImVec2 pos, ImVec2 size);
	void drawMapStage(ImVec2 pos, ImVec2 size);
	void drawTerrainPanel();
	void drawConditionsPanel();
	void drawSimulationPanel();
	void drawPlotsBand(ImVec2 pos, ImVec2 size);
	// panel pieces
	void drawSelectionHeader();
	void drawWaterTab();
	void drawSoilTab();
	void drawZonesTab();
	void drawBoundariesTab();
	void drawChronologyEditor();
	void drawTransportSection();
	void drawBalanceSection();
	void drawProbesSection();
	void drawDisplaySection();
	void drawAdvancedSection();
	void drawCameraButtons();
	void drawPerfWindow();
	void drawLegend();
	void drawProbeMarkers();
	void drawHelp();
	void drawProfilePlot();
	void updateProfileCells();
	std::optional<Probe> pickCell(glm::vec2 mousePixels) const;
	void                 updateProbeList();
	void                 autoColorRange();

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height);

	Options                            opts_;
	GLFWwindow                        *window_ = nullptr;
	std::unique_ptr<vkc::Context>      ctx_;
	std::unique_ptr<vkc::Swapchain>    swapchain_;
	std::unique_ptr<Session>           session_;
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
	bool          paused_        = true;        // everything starts paused: "Lancer" starts the run
	int           stepsPerFrame_ = 30;          // main.py: steps_per_render
	double        volume0_       = -1.0;        // initial volume (mass balance)
	double        balance0_      = 0.0;         // surface + aquifer - chronology inputs at the first readback (conserved)
	History       history_;

	// Georeferenced case (imports): grid origin and the orthophoto bytes (re-applied after bed edits)
	std::optional<geo::GridGeometry> caseGeo_;
	geo::Image                       orthoImage_;

	// Workflow
	Stage      stage_ = Stage::Map;
	Transition transition_;
	bool       meshPreference_ = false;        // "Maillage" of the Simulation stage (the Conditions stage always shows it)

	// View
	OrbitCamera    camera_;
	RenderSettings render_;
	float          zScale0_  = 2.0f;
	bool           autoRange_ = true;
	glm::vec4      viewRect_{0.0f};           // 3D view rectangle (framebuffer pixels)

	// Probes
	std::vector<Probe> probes_;

	// Cell selection in the 3D view (edit tool)
	bool                 selectTool_ = false;
	bool                 selecting_  = false;
	std::optional<Probe> selAnchor_, selCurrent_;
	// edit panel state
	int   waterMode_   = 0;          // 0 depth, 1 level, 2 add
	float waterValue_  = 1.0f;
	double levelValue_ = 0.0;        // "niveau initial sur tout le domaine" (m DNG)
	int   zoneKind_    = 0;          // 0 source, 1 sink, 2 level
	float zoneValue_   = 10.0f;
	char  zoneLabel_[48] = "";
	Session::SoilParams soil_;       // soil tab values (defaults of the case when the aquifer is on)
	// infiltration / rain chronologies (edit tools)
	int   chronoKind_  = 1;          // 0 infiltration, 1 rain
	float chronoValue_ = 50.0f;      // new zone value
	bool  chronoMmh_   = true;       // values in mm/h (else m3/s for the whole zone, as wolfgpu)

	// profile (Courbes > Profil): along x at row j, along y at column i, or from probe 1 to probe 2
	int                       profileMode_  = 0;
	int                       profileIndex_ = -1;        // row / column; -1 = middle of the domain
	std::vector<uint32_t>     profileCells_;             // cells sent to the solver
	std::vector<ProfilePoint> profile_;                  // read back each frame
	bool                      selectProfileTab_ = false;

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
	bool showPlots_ = true, showPerf_ = false, showHelp_ = false, showDemo_ = false;
	bool hideGui_ = false;        // F11: 3D view only
};

}        // namespace swe
