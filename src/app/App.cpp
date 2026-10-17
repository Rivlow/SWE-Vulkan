#include "app/App.h"

#include "app/Headless.h"
#include "app/Screenshot.h"
#include "util/Log.h"
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
	return "swe_settings.json";        // next to imgui.ini, in the working directory
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

	solver_   = std::make_unique<Solver>(*ctx_);
	effects_  = std::make_unique<Effects>(*ctx_, *solver_);
	renderer_ = std::make_unique<Renderer>(*ctx_, *solver_, *effects_, colorFormat_);
	renderer_->resize(swapchain_->extent);
	caseList_ = listCases(resolveAssetDir(opts_.assetDir));

	loadSettings();        // may change the case, steps per frame and display options
	loadCase(opts_.caseName);
	if (opts_.stepsPerFrame > 0)
		stepsPerFrame_ = opts_.stepsPerFrame;
	if (opts_.showMesh)
		render_.showMesh = true;
	if (opts_.zScale > 0)
		render_.z_scale = opts_.zScale;
	if (opts_.colorMode >= 0)
		render_.colorMode = static_cast<ColorMode>(std::clamp(opts_.colorMode, 0, 4));
	if (!opts_.probes.empty())
	{
		auto const &c = solver_->caseData();
		for (auto [i, j] : opts_.probes)
			if (probes_.size() < MAX_PROBES)
				probes_.push_back({uint32_t(std::clamp(i, 0, int(c.nx) - 1)), uint32_t(std::clamp(j, 0, int(c.ny) - 1))});
		updateProbeList();
	}
}

App::~App()
{
	if (ctx_)
		ctx_->device.waitIdle();
	for (auto &w : writers_)
		w.wait();
	if (solver_)
		saveSettings();
	shutdownImGui();
	// Vulkan objects must die before the window/GLFW.
	queries_ = nullptr;
	renderer_.reset();
	effects_.reset();
	solver_.reset();
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
	window_ = glfwCreateWindow(1600, 950, "Saint-Venant 2D - HLLC (Vulkan)", nullptr, nullptr);
	if (!window_)
		throw std::runtime_error("glfwCreateWindow failed");
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
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
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
	info.DescriptorPoolSize  = 64;
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
	auto data = makeCase(name, resolveAssetDir(opts_.assetDir));
	if (!opts_.scheme.empty())
		data.scheme = opts_.scheme;
	solver_->load(data);
	effects_->setCase(solver_->caseData());
	renderer_->setCase(solver_->caseData());
	opts_.caseName  = name;
	zScale0_        = static_cast<float>(data.z_scale_0);
	render_.z_scale = zScale0_;
	renderer_->homeCamera(camera_, zScale0_);
	probes_.clear();
	updateProbeList();
	resetSimulation();
}

void App::resetSimulation()
{
	solver_->reset();
	effects_->reset();
	scalars_   = {};
	prevSteps_ = 0;
	volume0_   = -1.0;
	history_.clear();
}

void App::History::clear()
{
	t.clear();
	volume.clear();
	wetArea.clear();
	dt.clear();
	for (auto *series : {&eta, &depth, &speed})
		for (auto &v : *series)
			v.clear();
}

void App::History::push(SolverScalars const &s, double cellArea, size_t probes)
{
	t.push_back(s.t);
	volume.push_back(s.volume);
	wetArea.push_back(s.wetCells * cellArea);
	dt.push_back(s.dt);
	for (size_t p = 0; p < MAX_PROBES; ++p)
	{
		bool on = p < probes;
		eta[p].push_back(on ? s.probes[p].eta : NAN);
		depth[p].push_back(on ? s.probes[p].h : NAN);
		speed[p].push_back(on ? std::hypot(s.probes[p].u, s.probes[p].v) : NAN);
	}
	if (t.size() > 40000)
		decimate();
}

void App::History::decimate()
{
	auto half = [](std::vector<double> &v) {
		size_t k = 0;
		for (size_t i = 0; i < v.size(); i += 2)
			v[k++] = v[i];
		v.resize(k);
	};
	for (auto *v : {&t, &volume, &wetArea, &dt})
		half(*v);
	for (auto *series : {&eta, &depth, &speed})
		for (auto &v : *series)
			half(v);
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
		drawGui();
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
		if (ImGui::IsKeyPressed(ImGuiKey_Space, false))
			paused_ = !paused_;
		if (ImGui::IsKeyPressed(ImGuiKey_F12, false))
			captureRequest_ = std::filesystem::path("captures") / ("swe_" + timestamp() + ".png");
		if (ImGui::IsKeyPressed(ImGuiKey_R, false) && io.KeyCtrl)
			resetSimulation();
		if (ImGui::IsKeyPressed(ImGuiKey_Home, false))
			camera_.reset();
		if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
			showHelp_ = !showHelp_;
	}

	// Camera, in the 3D view rectangle: normalised cursor with origin bottom-left (as in ui/camera.py)
	glm::vec4 r = viewRect_;
	if (r.z < 1.0f || r.w < 1.0f)
		r = glm::vec4(0.0f, 0.0f, std::max(io.DisplaySize.x, 1.0f), std::max(io.DisplaySize.y, 1.0f));
	glm::vec2 cursor((io.MousePos.x - r.x) / r.z, 1.0f - (io.MousePos.y - r.y) / r.w);
	bool      blocked = io.WantCaptureMouse;

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
	auto s = solver_->readback(frameIndex_);
	if (s.n_steps > 0 && s.n_steps >= prevSteps_)
	{
		double dtFrame = ImGui::GetIO().DeltaTime;
		if (dtFrame > 0)
			stepsPerSec_ = 0.9 * stepsPerSec_ + 0.1 * (s.n_steps - prevSteps_) / dtFrame;
		bool advanced = s.n_steps > prevSteps_;
		prevSteps_    = s.n_steps;
		scalars_      = s;
		if (volume0_ < 0.0)
			volume0_ = s.volume;
		if (advanced)
		{
			auto const &c = solver_->caseData();
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
	solver_->appendSteps(solverCmds, paused_ ? 0u : static_cast<uint32_t>(stepsPerFrame_), frameIndex_);

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
	renderer_->record(cb, frameIndex_, image, view, camera_, settings, scalars_.t);

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
		render_.colorMode  = static_cast<ColorMode>(std::clamp(j.value("color_mode", 0), 0, 4));
		autoRange_         = j.value("auto_range", autoRange_);
		showPlots_         = j.value("show_plots", showPlots_);
		showPerf_          = j.value("show_perf", showPerf_);
		showProbes_        = j.value("show_probes", showProbes_);
		effects_->setTurbidity(j.value("turbidity", false));
	}
	catch (std::exception const &e)
	{
		log::warn("{}: {}", settingsPath().string(), e.what());
	}
}

void App::saveSettings() const
{
	nlohmann::json j{{"case", solver_->caseData().name},
	                 {"steps_per_frame", stepsPerFrame_},
	                 {"streamlines", render_.showLines},
	                 {"mesh", render_.showMesh},
	                 {"color_mode", static_cast<int>(render_.colorMode)},
	                 {"auto_range", autoRange_},
	                 {"turbidity", effects_->turbidity()},
	                 {"show_plots", showPlots_},
	                 {"show_perf", showPerf_},
	                 {"show_probes", showProbes_}};
	std::ofstream(settingsPath()) << j.dump(2) << '\n';
}

}        // namespace swe
