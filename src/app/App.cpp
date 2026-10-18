#include "app/App.h"

#include "app/Headless.h"
#include "geo/Crs.h"
#include "geo/Tiles.h"
#include "sim/Import.h"
#include "util/Png.h"
#include "util/Log.h"
#include "util/Paths.h"
#include "vk/Pipelines.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <implot.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace swe {

namespace {

void checkVk(VkResult r)
{
	if (r != VK_SUCCESS)
		log::error("imgui vulkan backend: VkResult {}", static_cast<int>(r));
}

std::string timestamp()
{
	auto t = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
	return std::format("{:%Y%m%d_%H%M%S}", t);
}

std::filesystem::path settingsPath()
{
	return paths::dataDir() / "swe_settings.json";
}

}        // namespace

// ---------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------

App::App(Options const &opts) :
    opts_(opts)
{
	initWindow();
	ctx_ = std::make_unique<vkc::Context>(vkc::ContextOptions{.validation     = opts.validation,
	                                                          .syncValidation = opts.syncValidation,
	                                                          .forcedGpu      = opts.gpu,
	                                                          .window         = window_});
	swapchain_   = std::make_unique<vkc::Swapchain>(*ctx_, window_);
	colorFormat_ = swapchain_->surfaceFormat.format;

	auto cmds = vk::raii::CommandBuffers(ctx_->device, {.commandPool        = *ctx_->commandPool,
	                                                    .level              = vk::CommandBufferLevel::ePrimary,
	                                                    .commandBufferCount = 2 * kFramesInFlight});
	for (uint32_t i = 0; i < kFramesInFlight; ++i)
	{
		frames_[i].cmd            = std::move(cmds[2 * i]);
		frames_[i].begin          = std::move(cmds[2 * i + 1]);
		frames_[i].inFlight       = vk::raii::Fence(ctx_->device, {.flags = vk::FenceCreateFlagBits::eSignaled});
		frames_[i].imageAvailable = vk::raii::Semaphore(ctx_->device, vk::SemaphoreCreateInfo{});
	}
	createTimestamps();

	initImGui();

	session_  = std::make_unique<Session>(*ctx_);
	session_->solver().setAllTilesActive(opts_.allTiles);
	effects_  = std::make_unique<Effects>(*ctx_, session_->solver());
	renderer_ = std::make_unique<Renderer>(*ctx_, session_->solver(), *effects_, colorFormat_);
	renderer_->resize(swapchain_->extent);
	caseList_ = listCases(resolveAssetDir(opts_.assetDir));

	// geographic import
	geo::init();
	library_ = std::make_unique<geo::LocalLibrary>();
	for (auto const &d : opts_.libraryDirs)
		library_->addFolder(d);
	if (!opts_.libraryDirs.empty())
		library_->rescan();
	builder_ = std::make_unique<geo::TerrainBuilder>(*library_);
	map_     = std::make_unique<MapPanel>(*ctx_, *builder_);

	loadSettings();        // may change the case, steps per frame and display options
	loadCase(opts_.caseName);
	if (opts_.stepsPerFrame > 0)
		stepsPerFrame_ = opts_.stepsPerFrame;
	if (opts_.showMesh)
		render_.showMesh = true;
	if (opts_.gwContours > 0.0f)
	{
		render_.gwContours    = true;
		render_.gwContourStep = opts_.gwContours;
	}
	render_.gwXray = render_.gwXray || opts_.gwXray;
	selectProfileTab_ = opts_.showProfile;
	if (opts_.showProfile && opts_.probes.size() >= 2)
		profileMode_ = 2;        // probe 1 -> probe 2
	if (opts_.gwLines)
		effects_->setGroundwaterLines(true);
	if (opts_.crop)
	{
		render_.cropOn = true;
		render_.crop   = {opts_.crop->i0, opts_.crop->j0, opts_.crop->i1, opts_.crop->j1};
	}
	if (opts_.zScale > 0)
		render_.z_scale = opts_.zScale;
	if (opts_.colorMode >= 0)
		render_.colorMode = static_cast<ColorMode>(std::clamp(opts_.colorMode, 0, int(ColorMode::Count) - 1));
	hideGui_        = opts_.hideGui;
	meshPreference_ = render_.showMesh;
	applyScriptedEdits();

	// starting stage: the map, unless a case or a stage was asked for; --run starts simulating at once
	Stage start = opts_.caseExplicit ? Stage::Terrain : Stage::Map;
	if (opts_.stage >= 0)
		start = static_cast<Stage>(std::clamp(opts_.stage, 0, 3));
	if (opts_.run)
		start = Stage::Simulation;
	setStage(start);
	paused_ = !opts_.run;
	if (start == Stage::Terrain && renderer_->hasOrthophoto() && !opts_.topView)
		startTransition();        // the relief rises out of the map, as after an import
	if (opts_.topView)
		topView();
}

// ---------------------------------------------------------------------------------------------
// Workflow stages
// ---------------------------------------------------------------------------------------------

void App::setStage(Stage s)
{
	if (s != Stage::Map && !session_->loaded())
		return;
	Stage const previous = stage_;
	stage_               = s;
	if (s == Stage::Simulation && previous == Stage::Conditions)
		paused_ = false;        // "Lancer la simulation"
	if (s != Stage::Simulation)
		paused_ = true;
	if (previous == Stage::Simulation && s != Stage::Simulation)
		meshPreference_ = render_.showMesh;
	// the mesh shows where the conditions are placed, the selection tool is the main interaction there
	render_.showMesh = s == Stage::Conditions ? true : meshPreference_;
	selectTool_      = s == Stage::Conditions;
	if (!selectTool_)
	{
		selAnchor_.reset();
		selCurrent_.reset();
		render_.selection = {1, 0, 0, 0};
	}
	// back to the conditions: the run restarts from the initial state
	if (s == Stage::Conditions && previous == Stage::Simulation)
		resetSimulation();
	if (s == Stage::Map && map_ && caseGeo_)
		map_->showBox({caseGeo_->x0, caseGeo_->y0, caseGeo_->x0 + caseGeo_->nx * caseGeo_->cell, caseGeo_->y0 + caseGeo_->ny * caseGeo_->cell},
		              caseGeo_->epsg, previous == Stage::Map);
	if (previous == Stage::Map && s == Stage::Terrain && !transition_.active)
		camera_.reset();
	if (s == Stage::Conditions)
		soil_ = session_->caseData().gw.enabled ? session_->soilDefaults() : soil_;
}

// A saved case chosen in the Map stage: loaded, then shown in 3D (animated from the map when it has an orthophoto)
void App::openCase(std::string const &name)
{
	if (name != session_->caseData().name)
		loadCase(name);
	if (renderer_->hasOrthophoto())
		startTransition();
	else
		setStage(Stage::Terrain);
}

OrbitCamera::Pose App::topPose() const
{
	auto const &c = session_->caseData();
	float       S = renderer_->S();
	glm::vec3   focal(0.5f * float(c.Lx) * S, float(c.eta_0) * render_.z_scale * S, -0.5f * float(c.Ly) * S);
	return OrbitCamera::topPose(focal, std::max(float(c.Lx), float(c.Ly)) * S, camera_.fovY);
}

void App::topView()
{
	camera_.setPose(topPose());
}

void App::animateCamera(OrbitCamera::Pose const &target, float seconds)
{
	camera_.animateTo(target, seconds);
}

// Map -> Terrain: start flat, seen from above with the orthophoto, under the fading map
void App::startTransition()
{
	stage_             = Stage::Map;        // so that setStage sees the Map -> Terrain move
	transition_.active = true;
	transition_.t      = 0.0f;
	render_.z_scale    = 0.0f;
	render_.showOrtho  = true;
	render_.orthoMix   = 1.0f;
	map_->setFade(1.0f);
	// the map frames the footprint, so that the 3D top view appears where the box was
	if (caseGeo_)
		map_->showBox({caseGeo_->x0, caseGeo_->y0, caseGeo_->x0 + caseGeo_->nx * caseGeo_->cell, caseGeo_->y0 + caseGeo_->ny * caseGeo_->cell},
		              caseGeo_->epsg, true);
	camera_.setPose(topPose());
	setStage(Stage::Terrain);
}

void App::tickTransition(float dt)
{
	if (!transition_.active)
		return;
	transition_.t += dt;
	float t = transition_.t;
	map_->setFade(std::clamp(1.0f - t / kFadeSeconds, 0.0f, 1.0f));
	if (t >= kSwingStart && !camera_.animating() && t < kSwingStart + dt + 1e-3f)
		camera_.animateTo(camera_.home(), kSwingSeconds);
	float s = std::clamp((t - kSwingStart) / kSwingSeconds, 0.0f, 1.0f);
	render_.z_scale = zScale0_ * s * s * (3.0f - 2.0f * s);
	if (t >= kSwingStart + kSwingSeconds)
	{
		transition_.active = false;
		render_.z_scale    = zScale0_;
		map_->setFade(1.0f);
	}
}

// Edits given on the command line: water blocks and zones on cell rectangles, top view, probes
void App::applyScriptedEdits()
{
	auto const &c = session_->caseData();
	applyCaseEdits(*session_, opts_);
	if (opts_.topView)
		topView();
	if (!opts_.probes.empty())
	{
		for (auto [i, j] : opts_.probes)
			if (probes_.size() < MAX_PROBES)
				probes_.push_back({uint32_t(std::clamp(i, 0, int(c.nx) - 1)), uint32_t(std::clamp(j, 0, int(c.ny) - 1))});
		updateProbeList();
	}
}

CellRect App::selection() const
{
	auto s = render_.selection;
	return {s.x, s.y, s.z, s.w};
}

App::~App()
{
	if (ctx_)
		ctx_->device.waitIdle();
	for (auto &w : writers_)
		w.wait();
	if (session_ && session_->loaded())
		saveSettings();
	map_.reset();        // ImGui textures: freed while the backend is alive
	shutdownImGui();
	// Vulkan objects must die before the window/GLFW
	queries_ = nullptr;
	renderer_.reset();
	effects_.reset();
	session_.reset();
	for (auto &f : frames_)
		f = Frame{};
	swapchain_.reset();
	ctx_.reset();
	if (window_)
		glfwDestroyWindow(window_);
	glfwTerminate();
}

void App::initWindow()
{
	if (!glfwInit())
		throw std::runtime_error("glfwInit failed");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window_ = glfwCreateWindow(1600, 950, "SWE Vulkan", nullptr, nullptr);
	if (!window_)
		throw std::runtime_error("glfwCreateWindow failed");
	// window icon (the executable's resource covers the taskbar on Windows; this one covers every platform)
	std::ifstream f(std::filesystem::path(resolveAssetDir(opts_.assetDir)) / "icon.png", std::ios::binary);
	if (f)
	{
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
		auto                 icon = geo::decodeImage(bytes);
		if (!icon.empty())
		{
			GLFWimage image{int(icon.width), int(icon.height), icon.rgba.data()};
			glfwSetWindowIcon(window_, 1, &image);
		}
	}
	glfwSetWindowUserPointer(window_, this);
	glfwSetFramebufferSizeCallback(window_, &App::framebufferResizeCallback);
}

void App::framebufferResizeCallback(GLFWwindow *window, int, int)
{
	static_cast<App *>(glfwGetWindowUserPointer(window))->framebufferResized_ = true;
}

void App::initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImPlot::CreateContext();
	auto &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.IniFilename = nullptr;        // fixed layout: nothing to remember (and no stale docking from older versions)
	ImGui::StyleColorsDark();
	auto &style            = ImGui::GetStyle();
	style.WindowRounding   = 4.0f;
	style.FrameRounding    = 3.0f;
	style.GrabRounding     = 3.0f;
	style.WindowBorderSize = 0.0f;

	ImGui_ImplGlfw_InitForVulkan(window_, true);

	ImGui_ImplVulkan_InitInfo info{};
	info.ApiVersion          = VK_API_VERSION_1_4;
	info.Instance            = *ctx_->instance;
	info.PhysicalDevice      = *ctx_->physicalDevice;
	info.Device              = *ctx_->device;
	info.QueueFamily         = ctx_->queueFamily;
	info.Queue               = *ctx_->queue;
	info.DescriptorPoolSize  = 1024;        // map tiles are ImGui textures
	info.MinImageCount       = swapchain_->minImageCount;
	info.ImageCount          = swapchain_->imageCount();
	info.UseDynamicRendering = true;
	info.CheckVkResultFn     = &checkVk;

	// The GUI is drawn in a second, single-sampled pass straight onto the swapchain image.
	auto &pipe                                               = info.PipelineInfoMain;
	pipe.MSAASamples                                         = VK_SAMPLE_COUNT_1_BIT;
	pipe.PipelineRenderingCreateInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	pipe.PipelineRenderingCreateInfo.colorAttachmentCount    = 1;
	pipe.PipelineRenderingCreateInfo.pColorAttachmentFormats = reinterpret_cast<VkFormat const *>(&colorFormat_);

	if (!ImGui_ImplVulkan_Init(&info))
		throw std::runtime_error("ImGui_ImplVulkan_Init failed");
	imguiReady_ = true;
}

void App::shutdownImGui()
{
	if (!imguiReady_)
		return;
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImPlot::DestroyContext();
	ImGui::DestroyContext();
	imguiReady_ = false;
}

// GPU timestamps: per frame slot, [reset + T0] in a static "begin" command buffer submitted before the
// solver steps, T1..T3 in the frame command buffer.
void App::createTimestamps()
{
	auto family = ctx_->physicalDevice.getQueueFamilyProperties()[ctx_->queueFamily];
	if (family.timestampValidBits == 0 || ctx_->properties.limits.timestampPeriod <= 0.0f)
		return;
	timestampPeriod_ = ctx_->properties.limits.timestampPeriod;
	queries_ = vk::raii::QueryPool(ctx_->device, {.queryType = vk::QueryType::eTimestamp, .queryCount = kFramesInFlight * kTimestamps});
	for (uint32_t s = 0; s < kFramesInFlight; ++s)
	{
		auto &cb = frames_[s].begin;
		cb.begin({});
		cb.resetQueryPool(*queries_, s * kTimestamps, kTimestamps);
		cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *queries_, s * kTimestamps);
		cb.end();
	}
}

void App::readTimestamps(uint32_t slot)
{
	if (!*queries_ || frameCount_ < kFramesInFlight)
		return;
	auto [r, ts] = queries_.getResults<uint64_t>(slot * kTimestamps, kTimestamps, kTimestamps * sizeof(uint64_t),
	                                             sizeof(uint64_t), vk::QueryResultFlagBits::e64);
	if (r != vk::Result::eSuccess || ts.size() < kTimestamps)
		return;
	auto ms = [&](int a, int b) { return double(ts[b] - ts[a]) * timestampPeriod_ * 1e-6; };
	auto smooth = [](double &acc, double v) { acc = acc == 0.0 ? v : 0.9 * acc + 0.1 * v; };
	smooth(gpu_.solver, ms(0, 1));
	smooth(gpu_.effects, ms(1, 2));
	smooth(gpu_.render, ms(2, 3));
}

// ---------------------------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------------------------

void App::loadCase(std::string const &name)
{
	auto assets = resolveAssetDir(opts_.assetDir);
	session_->load(makeCase(name, assets, opts_.caseOverrides()));
	opts_.caseName = name;
	caseGeo_.reset();
	orthoImage_ = {};
	onCaseLoaded();
	auto const &data = session_->caseData();
	zScale0_        = static_cast<float>(data.z_scale_0);
	render_.z_scale = zScale0_;
	renderer_->homeCamera(camera_, zScale0_);
	probes_.clear();
	profileCells_.clear();        // the solver forgot them with the old grid
	profileIndex_ = -1;
	render_.cropOn = false;
	updateProbeList();
	// georeferenced cases: footprint on the map, orthophoto drape, bed tools
	auto origin = caseOrigin(name, assets);
	if (origin.valid)
	{
		caseGeo_ = geo::GridGeometry{data.nx, data.ny, origin.x0, origin.y0, origin.cell, origin.epsg};
		if (map_)
			map_->showBox({origin.x0, origin.y0, origin.x0 + data.nx * origin.cell, origin.y0 + data.ny * origin.cell}, origin.epsg, true);
	}
	if (map_)
		map_->setCaseGeometry(caseGeo_);
	if (!origin.orthoFile.empty())
	{
		std::ifstream f(std::filesystem::path(assets) / origin.orthoFile, std::ios::binary);
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
		orthoImage_ = geo::decodeImage(bytes);
		if (!orthoImage_.empty())
			renderer_->setOrthophoto(orthoImage_.width, orthoImage_.height, orthoImage_.rgba.data());
		else
			log::warn("orthophoto {} not readable", origin.orthoFile);
	}
}

// The session (re)loaded its case: views follow, clock and plots restart.
void App::onCaseLoaded()
{
	effects_->setCase(session_->caseData());
	renderer_->setCase(session_->caseData());
	if (!orthoImage_.empty())
		renderer_->setOrthophoto(orthoImage_.width, orthoImage_.height, orthoImage_.rgba.data());
	effects_->reset();
	scalars_   = {};
	prevSteps_ = 0;
	volume0_   = -1.0;
	history_.clear();
}

// Bed edit from the map: modifies the loaded case in memory (Save keeps it).
void App::applyBedEdit(BedEdit const &edit)
{
	if (!caseGeo_)
		return;
	auto   zb = session_->caseData().zb;
	size_t n  = geo::applyBedEdit(zb, *caseGeo_, edit);
	log::info("bed edit: {} cell(s) modified", n);
	if (n == 0)
		return;
	session_->replaceBed(std::move(zb));
	onCaseLoaded();
	updateProbeList();
	profileCells_.clear();        // the reload forgot them
}

// Requests of the map panel (finished import, bed edit), whatever the stage shown
void App::processMapRequests()
{
	map_->pollJob();
	if (auto outcome = map_->takeOutcome())
		loadImport(std::move(*outcome));
	if (auto edit = map_->takeBedEdit())
		applyBedEdit(*edit);
}

void App::saveCurrentCase()
{
	if (!caseGeo_)
		return;
	updateImportedCase(session_->caseData(), resolveAssetDir(opts_.assetDir));
	caseList_ = listCases(resolveAssetDir(opts_.assetDir));
	log::info("case '{}' saved", session_->caseData().name);
}

// A finished geographic import: saved as a case (JSON + npz + orthophoto), then loaded like any other.
void App::loadImport(ImportOutcome &&o)
{
	auto assets = resolveAssetDir(opts_.assetDir);
	auto data   = saveImport(o.result, o.request, o.orthophoto, assets);
	caseList_   = listCases(assets);
	loadCase(data.name);
	log::info("imported case '{}' ({} x {} cells at {} m)", data.name, data.nx, data.ny, o.result.bed.cell);
	if (renderer_->hasOrthophoto())
		startTransition();
	else
		setStage(Stage::Terrain);
}

void App::resetSimulation()
{
	session_->reset();
	effects_->reset();
	scalars_   = {};
	prevSteps_ = 0;
	volume0_   = -1.0;
	history_.clear();
}

// ---------------------------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------------------------

void App::run()
{
	if (!opts_.capturePath.empty())
		log::info("capture: {} after {} frames", opts_.capturePath, opts_.captureFrames);

	while (!glfwWindowShouldClose(window_))
	{
		glfwPollEvents();
		if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED))
		{
			glfwWaitEvents();
			continue;
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		float dt = std::min(ImGui::GetIO().DeltaTime, 0.1f);
		tickTransition(dt);
		camera_.tick(dt);
		drawGui();
		processMapRequests();
		handleInput();
		ImGui::Render();

		if (!opts_.capturePath.empty() && frameCount_ == static_cast<uint64_t>(opts_.captureFrames))
			captureRequest_ = opts_.capturePath;

		drawFrame();
		++frameCount_;

		if (!opts_.capturePath.empty() && captureDone_)
			glfwSetWindowShouldClose(window_, GLFW_TRUE);
	}
	ctx_->device.waitIdle();
	for (auto &f : frames_)
		if (f.capturePath)
			saveCapture(f);
}

void App::handleInput()
{
	auto &io = ImGui::GetIO();
	if (!io.WantCaptureKeyboard)
	{
		if (stage_ == Stage::Simulation && ImGui::IsKeyPressed(ImGuiKey_Space, false))
			paused_ = !paused_;
		if (ImGui::IsKeyPressed(ImGuiKey_F12, false))
			captureRequest_ = paths::dataDir() / "captures" / ("swe_" + timestamp() + ".png");
		if (stage_ == Stage::Simulation && ImGui::IsKeyPressed(ImGuiKey_R, false))
			resetSimulation();
		if (ImGui::IsKeyPressed(ImGuiKey_Home, false))
			animateCamera(camera_.home());
		if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
			showHelp_ = !showHelp_;
		if (ImGui::IsKeyPressed(ImGuiKey_F11, false))
			hideGui_ = !hideGui_;
	}
	// the map stage has no 3D interaction; during the transition the camera is scripted (a click ends it)
	if (stage_ == Stage::Map)
		return;
	if (transition_.active)
	{
		if (!io.WantCaptureMouse && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)))
		{
			transition_.active = false;
			render_.z_scale    = zScale0_;
			map_->setFade(1.0f);
			camera_.stop();
		}
		return;
	}

	// Camera, in the 3D view rectangle: normalised cursor with origin bottom-left (as in ui/camera.py)
	glm::vec4 r = viewRect_;
	if (r.z < 1.0f || r.w < 1.0f)
		r = glm::vec4(0.0f, 0.0f, std::max(io.DisplaySize.x, 1.0f), std::max(io.DisplaySize.y, 1.0f));
	glm::vec2 cursor((io.MousePos.x - r.x) / r.z, 1.0f - (io.MousePos.y - r.y) / r.w);
	bool      blocked = io.WantCaptureMouse;

	// Cell selection tool: left drag = rectangle of cells (camera untouched)
	if (selectTool_)
	{
		if (!blocked && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
		{
			selAnchor_  = pickCell({io.MousePos.x, io.MousePos.y});
			selCurrent_ = selAnchor_;
			selecting_  = selAnchor_.has_value();
		}
		if (selecting_ && ImGui::IsMouseDown(ImGuiMouseButton_Left))
			if (auto c = pickCell({io.MousePos.x, io.MousePos.y}))
				selCurrent_ = c;
		if (selecting_ && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
			selecting_ = false;
		if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Escape, false))
		{
			selAnchor_.reset();
			selCurrent_.reset();
		}
		if (selAnchor_ && selCurrent_)
			render_.selection = {std::min(selAnchor_->i, selCurrent_->i), std::min(selAnchor_->j, selCurrent_->j),
			                     std::max(selAnchor_->i, selCurrent_->i), std::max(selAnchor_->j, selCurrent_->j)};
		else
			render_.selection = {1, 0, 0, 0};
		// wheel zoom and middle pan still work, no rotation
		int button = ImGui::IsMouseDown(ImGuiMouseButton_Middle) ? 2 : 0;
		camera_.update(cursor, button, blocked ? 0.0f : io.MouseWheel, blocked);
		return;
	}

	// Ctrl + click on the terrain: new probe (camera untouched)
	if (!blocked && io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
	{
		if (auto cell = pickCell({io.MousePos.x, io.MousePos.y}))
		{
			if (probes_.size() >= MAX_PROBES)
				probes_.erase(probes_.begin());
			probes_.push_back(*cell);
			updateProbeList();
		}
		return;
	}
	int button = ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyCtrl ? 1 : ImGui::IsMouseDown(ImGuiMouseButton_Middle) ? 2 : 0;
	camera_.update(cursor, button, blocked ? 0.0f : io.MouseWheel, blocked);
}

void App::drawFrame()
{
	auto &frame = frames_[frameIndex_];
	if (ctx_->device.waitForFences(*frame.inFlight, vk::True, UINT64_MAX) != vk::Result::eSuccess)
		throw std::runtime_error("waitForFences failed");

	// The fence covered the solver batch of this slot too: its scalar readback is complete.
	auto s = session_->solver().readback(frameIndex_);
	profile_ = session_->solver().readProfile(frameIndex_);
	// (the diagnostics run every frame, paused at t = 0 included: the balance is right before "Lancer")
	if (s.n_steps >= prevSteps_)
	{
		double dtFrame = ImGui::GetIO().DeltaTime;
		if (dtFrame > 0)
			stepsPerSec_ = 0.9 * stepsPerSec_ + 0.1 * (s.n_steps - prevSteps_) / dtFrame;
		bool advanced = s.n_steps > prevSteps_;
		prevSteps_    = s.n_steps;
		scalars_      = s;
		if (volume0_ < 0.0)
		{
			balance0_ = double(s.volume) + s.gwVolume - s.srcNet;
			volume0_ = s.volume;
		}
		if (advanced)
		{
			auto const &c = session_->caseData();
			history_.push(s, c.dx() * c.dy(), probes_.size());
		}
	}
	readTimestamps(frameIndex_);
	if (frame.capturePath)
		saveCapture(frame);

	auto [acquire, imageIndex] = swapchain_->swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable, nullptr);
	if (acquire == vk::Result::eErrorOutOfDateKHR)
	{
		swapchain_->recreate();
		onSwapchainRecreated();
		return;
	}
	ctx_->device.resetFences(*frame.inFlight);

	recordFrame(frame, imageIndex);

	// One submission, two batches, one fence: [timestamp, N solver steps, diagnostics + readback] then [frame].
	std::vector<vk::CommandBufferSubmitInfo> solverCmds;
	if (*queries_)
		solverCmds.push_back({.commandBuffer = *frame.begin});
	session_->solver().appendSteps(solverCmds, paused_ ? 0u : static_cast<uint32_t>(stepsPerFrame_), frameIndex_);

	vk::SemaphoreSubmitInfo     waitInfo{.semaphore = *frame.imageAvailable, .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput};
	vk::SemaphoreSubmitInfo     signalInfo{.semaphore = *swapchain_->renderFinished[imageIndex], .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
	vk::CommandBufferSubmitInfo cbInfo{.commandBuffer = *frame.cmd};
	std::array<vk::SubmitInfo2, 2> batches{
	    vk::SubmitInfo2{.commandBufferInfoCount = static_cast<uint32_t>(solverCmds.size()), .pCommandBufferInfos = solverCmds.data()},
	    vk::SubmitInfo2{.waitSemaphoreInfoCount   = 1,
	                    .pWaitSemaphoreInfos      = &waitInfo,
	                    .commandBufferInfoCount   = 1,
	                    .pCommandBufferInfos      = &cbInfo,
	                    .signalSemaphoreInfoCount = 1,
	                    .pSignalSemaphoreInfos    = &signalInfo}};
	ctx_->queue.submit2(batches, *frame.inFlight);

	vk::PresentInfoKHR presentInfo{.waitSemaphoreCount = 1,
	                               .pWaitSemaphores    = &*swapchain_->renderFinished[imageIndex],
	                               .swapchainCount     = 1,
	                               .pSwapchains        = &*swapchain_->swapchain,
	                               .pImageIndices      = &imageIndex};
	auto               present = ctx_->queue.presentKHR(presentInfo);
	if (present == vk::Result::eErrorOutOfDateKHR || present == vk::Result::eSuboptimalKHR || framebufferResized_)
	{
		framebufferResized_ = false;
		swapchain_->recreate();
		onSwapchainRecreated();
	}

	frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
}

void App::recordFrame(Frame &frame, uint32_t imageIndex)
{
	auto &cb    = frame.cmd;
	auto  image = swapchain_->images[imageIndex];
	auto  view  = *swapchain_->views[imageIndex];
	auto  q0    = frameIndex_ * kTimestamps;
	cb.reset();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
	if (*queries_)
		cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *queries_, q0 + 1);

	// effects of this frame (they read the state the solver batch has just produced)
	double tFrame = paused_ ? 0.0 : double(stepsPerFrame_) * scalars_.dt;        // estimate (substep count only)
	float  realDt = paused_ ? 0.0f : std::min(ImGui::GetIO().DeltaTime, 0.1f);
	bool   lines  = render_.showLines && render_.showWater;
	effects_->record(cb, EffectsFrame{.zScale      = render_.z_scale,
	                                  .realDt      = realDt,
	                                  .substeps    = effects_->substeps(tFrame, scalars_.s_max),
	                                  .streamlines = lines});
	if (*queries_)
		cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *queries_, q0 + 2);

	// 3D scene (multisampled, resolved into the swapchain image), in the central dock area
	RenderSettings settings = render_;
	settings.showLines      = lines;
	settings.viewport       = viewRect_;
	renderer_->record(cb, frameIndex_, image, view, camera_, settings, static_cast<float>(scalars_.t));

	// GUI on top, single-sampled
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
	                   vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	                   vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite);
	vk::RenderingAttachmentInfo color{.imageView   = view,
	                                  .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
	                                  .loadOp      = vk::AttachmentLoadOp::eLoad,
	                                  .storeOp     = vk::AttachmentStoreOp::eStore};
	cb.beginRendering(vk::RenderingInfo{.renderArea           = {.offset = {0, 0}, .extent = swapchain_->extent},
	                                    .layerCount           = 1,
	                                    .colorAttachmentCount = 1,
	                                    .pColorAttachments    = &color});
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cb);
	cb.endRendering();

	// screenshot or recording frame
	std::optional<std::filesystem::path> shot = captureRequest_;
	captureRequest_.reset();
	if (!shot && recording_ && !paused_ && frameCount_ % std::max(recordEvery_, 1) == 0)
		shot = recordDir_ / std::format("frame_{:05d}.png", recordIndex_++);
	if (shot)
	{
		recordCapture(frame, cb, image);
		frame.capturePath = shot;
	}
	else
		vkc::transitionImage(cb, image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
		                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
		                     vk::PipelineStageFlagBits2::eBottomOfPipe, {});
	if (*queries_)
		cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *queries_, q0 + 3);
	cb.end();
}

void App::recordCapture(Frame &frame, vk::raii::CommandBuffer const &cb, vk::Image image)
{
	auto           extent = swapchain_->extent;
	vk::DeviceSize size   = vk::DeviceSize(extent.width) * extent.height * 4;
	if (!frame.capture || frame.capture.size() < size)
		frame.capture = vkc::Buffer::host(*ctx_, size, vk::BufferUsageFlagBits::eTransferDst, "capture");
	frame.captureExtent = extent;

	vkc::transitionImage(cb, image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
	                     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	cb.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, frame.capture.handle(),
	                     vk::BufferImageCopy{.bufferOffset     = 0,
	                                         .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	                                         .imageExtent      = {extent.width, extent.height, 1}});
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	                   vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
	vkc::transitionImage(cb, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::ePresentSrcKHR,
	                     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
	                     vk::PipelineStageFlagBits2::eBottomOfPipe, {});
}

// Called once the fence of the frame that recorded the copy has signalled: copies the pixels and hands
// the PNG encoding to a worker thread (recording must not stall the frame loop).
void App::saveCapture(Frame &frame)
{
	auto     path   = *frame.capturePath;
	auto     extent = frame.captureExtent;
	bool     bgra   = colorFormat_ == vk::Format::eB8G8R8A8Unorm || colorFormat_ == vk::Format::eB8G8R8A8Srgb;
	auto     bytes  = static_cast<uint8_t const *>(frame.capture.mapped());
	std::vector<uint8_t> pixels(bytes, bytes + size_t(extent.width) * extent.height * 4);
	frame.capturePath.reset();

	bool single = !recording_ || path.parent_path() != recordDir_;
	writers_.push_back(std::async(std::launch::async, [path, extent, bgra, single, pixels = std::move(pixels)] {
		bool ok = writePng(path, pixels.data(), extent.width, extent.height, bgra);
		if (ok && single)
			log::info("capture saved: {}", std::filesystem::absolute(path).string());
		return ok;
	}));
	std::erase_if(writers_, [](auto &f) { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; });
	if (!opts_.capturePath.empty() && path == std::filesystem::path(opts_.capturePath))
	{
		for (auto &w : writers_)
			w.wait();
		captureDone_ = true;
	}
}

void App::onSwapchainRecreated()
{
	ImGui_ImplVulkan_SetMinImageCount(swapchain_->minImageCount);
	if (swapchain_->surfaceFormat.format != colorFormat_)
		throw std::runtime_error("swapchain format changed on recreation (unsupported)");
	renderer_->resize(swapchain_->extent);
}

// ---------------------------------------------------------------------------------------------
// Settings persistence (swe_settings.json, next to imgui.ini)
// ---------------------------------------------------------------------------------------------

void App::loadSettings()
{
	std::ifstream f(settingsPath());
	if (!f)
		return;
	try
	{
		auto j = nlohmann::json::parse(f);
		if (!opts_.caseExplicit && j.contains("case"))
		{
			auto name = j["case"].get<std::string>();
			if (std::ranges::any_of(caseList_, [&](CaseInfo const &c) { return c.name == name; }))
				opts_.caseName = name;
		}
		stepsPerFrame_     = std::clamp(j.value("steps_per_frame", stepsPerFrame_), 1, 500);
		render_.showLines  = j.value("streamlines", render_.showLines);
		render_.showMesh   = j.value("mesh", render_.showMesh);
		render_.colorMode  = static_cast<ColorMode>(std::clamp(j.value("color_mode", 0), 0, int(ColorMode::Count) - 1));
		autoRange_         = j.value("auto_range", autoRange_);
		showPlots_         = j.value("show_plots", showPlots_);
		effects_->setTurbidity(j.value("turbidity", false));
		if (j.contains("map") && map_)
		{
			auto const &m = j["map"];
			map_->setView({m.value("lon", 5.57), m.value("lat", 50.63)}, m.value("zoom", 12.0));
			map_->setLayer(m.value("layer", 0));
		}
	}
	catch (std::exception const &e)
	{
		log::warn("{}: {}", settingsPath().string(), e.what());
	}
}

void App::saveSettings() const
{
	nlohmann::json j{{"case", session_->caseData().name},
	                 {"steps_per_frame", stepsPerFrame_},
	                 {"streamlines", render_.showLines},
	                 {"mesh", stage_ == Stage::Simulation ? render_.showMesh : meshPreference_},
	                 {"color_mode", static_cast<int>(render_.colorMode)},
	                 {"auto_range", autoRange_},
	                 {"turbidity", effects_->turbidity()},
	                 {"show_plots", showPlots_}};
	if (map_)
		j["map"] = {{"lon", map_->center().lon}, {"lat", map_->center().lat}, {"zoom", map_->zoom()}, {"layer", map_->layer()}};
	std::ofstream(settingsPath()) << j.dump(2) << '\n';
}

}        // namespace swe
