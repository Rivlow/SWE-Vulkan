#include "app/App.h"

#include "app/Headless.h"
#include "app/Screenshot.h"
#include "cases/Cases.h"
#include "util/Log.h"
#include "vk/Pipelines.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <implot.h>

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace swe {

namespace {

void checkVk(VkResult r)
{
	if (r != VK_SUCCESS)
		log::error("imgui vulkan backend: VkResult {}", static_cast<int>(r));
}

std::filesystem::path timestampedCapturePath()
{
	auto now = std::chrono::system_clock::now();
	auto t   = std::chrono::floor<std::chrono::seconds>(now);
	return std::filesystem::path("captures") / std::format("swe_{:%Y%m%d_%H%M%S}.png", t);
}

}        // namespace

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
	                                                    .commandBufferCount = kFramesInFlight});
	for (uint32_t i = 0; i < kFramesInFlight; ++i)
	{
		frames_[i].cmd            = std::move(cmds[i]);
		frames_[i].inFlight       = vk::raii::Fence(ctx_->device, {.flags = vk::FenceCreateFlagBits::eSignaled});
		frames_[i].imageAvailable = vk::raii::Semaphore(ctx_->device, vk::SemaphoreCreateInfo{});
	}

	initImGui();

	solver_   = std::make_unique<Solver>(*ctx_);
	renderer_ = std::make_unique<Renderer>(*ctx_, *solver_, colorFormat_);
	renderer_->resize(swapchain_->extent);
	loadCase(opts_.caseName);
	if (opts_.stepsPerFrame > 0)
		stepsPerFrame_ = opts_.stepsPerFrame;
	render_.showMesh = opts_.showMesh;
	if (opts_.zScale > 0)
		render_.z_scale = opts_.zScale;
}

App::~App()
{
	if (ctx_)
		ctx_->device.waitIdle();
	shutdownImGui();
	// Vulkan objects must die before the window/GLFW.
	captureBuffer_ = {};
	renderer_.reset();
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
	window_ = glfwCreateWindow(1280, 800, "Saint-Venant 2D - HLLC (Vulkan)", nullptr, nullptr);
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

void App::loadCase(std::string const &name)
{
	auto data = makeCase(name, resolveAssetDir(opts_.assetDir));
	if (!opts_.scheme.empty())
		data.scheme = opts_.scheme;
	solver_->load(data);
	renderer_->setCase(solver_->caseData());
	zScale0_        = static_cast<float>(data.z_scale_0);
	render_.z_scale = zScale0_;
	renderer_->homeCamera(camera_, zScale0_);
	scalars_   = {};
	prevSteps_ = 0;
}

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
		handleInput();
		drawGui();
		ImGui::Render();

		if (!opts_.capturePath.empty() && frameCount_ == static_cast<uint64_t>(opts_.captureFrames))
			captureRequest_ = opts_.capturePath;

		drawFrame();
		++frameCount_;

		// --capture: quit once the image has been written
		if (!opts_.capturePath.empty() && frameCount_ > static_cast<uint64_t>(opts_.captureFrames) && !capturePending_ &&
		    !captureRequest_)
			glfwSetWindowShouldClose(window_, GLFW_TRUE);
	}
	ctx_->device.waitIdle();
	if (capturePending_)
		saveCapture();
}

void App::handleInput()
{
	auto &io = ImGui::GetIO();
	if (!io.WantCaptureKeyboard)
	{
		if (ImGui::IsKeyPressed(ImGuiKey_Space, false))
			paused_ = !paused_;
		if (ImGui::IsKeyPressed(ImGuiKey_F12, false))
			captureRequest_ = timestampedCapturePath();
	}

	// Camera: normalised cursor with origin bottom-left (as in ui/camera.py); no drag starting on the GUI
	glm::vec2 size(std::max(io.DisplaySize.x, 1.0f), std::max(io.DisplaySize.y, 1.0f));
	glm::vec2 cursor(io.MousePos.x / size.x, 1.0f - io.MousePos.y / size.y);
	int       button = ImGui::IsMouseDown(ImGuiMouseButton_Left) ? 1 : ImGui::IsMouseDown(ImGuiMouseButton_Middle) ? 2 : 0;
	bool      blocked = io.WantCaptureMouse;
	camera_.update(cursor, button, blocked ? 0.0f : io.MouseWheel, blocked);
}

void App::drawGui()
{
	ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
	drawControls();
	if (showDemo_)
		ImGui::ShowDemoWindow(&showDemo_);
}

void App::drawControls()
{
	auto &io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360, 470), ImGuiCond_FirstUseEver);
	if (!ImGui::Begin("Contrôles"))
	{
		ImGui::End();
		return;
	}

	auto const &c = solver_->caseData();
	ImGui::Text("t = %.3f s   (%u pas)", scalars_.t, scalars_.n_steps);
	ImGui::Text("dt = %.2e s   s_max = %.3f m/s", scalars_.dt, scalars_.s_max);
	ImGui::Text("%.0f pas/s   %.1f fps", stepsPerSec_, io.Framerate);
	if (!std::isfinite(scalars_.t) || !std::isfinite(scalars_.dt))
		ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.3f, 1.0f), "La simulation a divergé : Reset simulation");

	ImGui::Separator();
	ImGui::Checkbox("Pause  (Espace)", &paused_);
	ImGui::SliderInt("Pas / rendu", &stepsPerFrame_, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp);
	if (ImGui::Button("Reset simulation"))
		resetSimulation();
	ImGui::SameLine();
	if (ImGui::Button("Reset caméra"))
		camera_.reset();

	ImGui::SeparatorText("Affichage");
	ImGui::Checkbox("Eau", &render_.showWater);
	ImGui::SameLine();
	ImGui::Checkbox("Maillage", &render_.showMesh);
	ImGui::SliderFloat("Exagération z", &render_.z_scale, 1.0f, 2.5f * zScale0_, "%.2f");
	ImGui::TextDisabled("Clic gauche : rotation, molette : zoom,\nclic milieu : translation, F12 : capture");

	ImGui::SeparatorText("Schéma numérique");
	if (ImGui::BeginCombo("Runge-Kutta", schemes().at(solver_->schemeName()).label.c_str()))
	{
		for (auto const &[name, tableau] : schemes())
		{
			bool selected = name == solver_->schemeName();
			if (ImGui::Selectable(tableau.label.c_str(), selected) && !selected)
				solver_->setScheme(name);
			if (selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	float cfl = static_cast<float>(c.cfl);
	if (ImGui::SliderFloat("CFL", &cfl, 0.05f, 0.9f, "%.2f"))
		solver_->setCfl(cfl);
	float manning = static_cast<float>(c.manning_n);
	if (ImGui::SliderFloat("Manning n", &manning, 0.0f, 0.1f, "%.3f"))
		solver_->setManning(manning);

	ImGui::SeparatorText("Cas");
	if (ImGui::BeginCombo("##case", c.name.c_str()))
	{
		for (auto const &name : caseNames())
			if (ImGui::Selectable(name.c_str(), name == c.name) && name != c.name)
				loadCase(name);
		ImGui::EndCombo();
	}
	ImGui::Text("%u x %u mailles (%.3g m)", c.nx, c.ny, c.dx());
	ImGui::TextDisabled("%s", ctx_->properties.deviceName.data());
	ImGui::Checkbox("Démo ImGui", &showDemo_);
	ImGui::End();
}

void App::resetSimulation()
{
	solver_->reset();
	scalars_   = {};
	prevSteps_ = 0;
}

void App::drawFrame()
{
	auto &frame = frames_[frameIndex_];
	if (ctx_->device.waitForFences(*frame.inFlight, vk::True, UINT64_MAX) != vk::Result::eSuccess)
		throw std::runtime_error("waitForFences failed");

	// The fence covered the solver batch of this slot too: its scalar readback is complete.
	auto s = solver_->readback(frameIndex_);
	if (s.n_steps >= prevSteps_ && s.n_steps > 0)
	{
		double dtFrame = ImGui::GetIO().DeltaTime;
		if (dtFrame > 0)
			stepsPerSec_ = 0.9 * stepsPerSec_ + 0.1 * (s.n_steps - prevSteps_) / dtFrame;
		prevSteps_ = s.n_steps;
		scalars_   = s;
	}
	if (capturePending_)
		saveCapture();

	auto [acquire, imageIndex] = swapchain_->swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable, nullptr);
	if (acquire == vk::Result::eErrorOutOfDateKHR)
	{
		swapchain_->recreate();
		onSwapchainRecreated();
		return;
	}
	ctx_->device.resetFences(*frame.inFlight);

	recordFrame(frame, imageIndex);

	// One submission, two batches, one fence: [N solver steps + scalar readback] then [frame rendering].
	std::vector<vk::CommandBufferSubmitInfo> solverCmds;
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
	cb.reset();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	// 3D scene (multisampled, resolved into the swapchain image)
	renderer_->record(cb, frameIndex_, image, view, camera_, render_, scalars_.t);

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

	if (captureRequest_)
	{
		recordCapture(cb, image);
		capturePending_ = captureRequest_;
		captureRequest_.reset();
	}
	else
		vkc::transitionImage(cb, image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
		                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
		                     vk::PipelineStageFlagBits2::eBottomOfPipe, {});
	cb.end();
}

void App::recordCapture(vk::raii::CommandBuffer const &cb, vk::Image image)
{
	auto           extent = swapchain_->extent;
	vk::DeviceSize size   = vk::DeviceSize(extent.width) * extent.height * 4;
	if (!captureBuffer_ || captureBuffer_.size() < size)
		captureBuffer_ = vkc::Buffer::host(*ctx_, size, vk::BufferUsageFlagBits::eTransferDst, "capture");
	captureExtent_ = extent;

	vkc::transitionImage(cb, image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
	                     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	cb.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, captureBuffer_.handle(),
	                     vk::BufferImageCopy{.bufferOffset      = 0,
	                                         .imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
	                                         .imageExtent       = {extent.width, extent.height, 1}});
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	                   vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
	vkc::transitionImage(cb, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::ePresentSrcKHR,
	                     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
	                     vk::PipelineStageFlagBits2::eBottomOfPipe, {});
}

void App::saveCapture()
{
	// called once the fence of the frame that recorded the copy has signalled (or after waitIdle)
	ctx_->device.waitIdle();
	bool bgra = colorFormat_ == vk::Format::eB8G8R8A8Unorm || colorFormat_ == vk::Format::eB8G8R8A8Srgb;
	if (writePng(*capturePending_, static_cast<uint8_t const *>(captureBuffer_.mapped()), captureExtent_.width,
	             captureExtent_.height, bgra))
		log::info("capture saved: {}", std::filesystem::absolute(*capturePending_).string());
	capturePending_.reset();
}

void App::onSwapchainRecreated()
{
	ImGui_ImplVulkan_SetMinImageCount(swapchain_->minImageCount);
	if (swapchain_->surfaceFormat.format != colorFormat_)
		throw std::runtime_error("swapchain format changed on recreation (unsupported)");
	renderer_->resize(swapchain_->extent);
}

}        // namespace swe
