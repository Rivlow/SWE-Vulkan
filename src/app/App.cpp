#include "app/App.h"

#include "app/Headless.h"
#include "cases/Cases.h"
#include "util/Log.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <implot.h>

#include <stdexcept>

namespace swe {

namespace {

// Background of the Python viewer (canvas.set_background_color).
constexpr std::array<float, 4> kSkyBackground{0.80f, 0.87f, 0.93f, 1.0f};

void checkVk(VkResult r)
{
	if (r != VK_SUCCESS)
		log::error("imgui vulkan backend: VkResult {}", static_cast<int>(r));
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

	auto data = makeCase(opts_.caseName, resolveAssetDir(opts_.assetDir));
	if (!opts_.scheme.empty())
		data.scheme = opts_.scheme;
	solver_ = std::make_unique<Solver>(*ctx_);
	solver_->load(data);
}

App::~App()
{
	if (ctx_)
		ctx_->device.waitIdle();
	shutdownImGui();
	// Vulkan objects must die before the window/GLFW.
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

void App::run()
{
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
		ImGui::Render();

		drawFrame();
	}
	ctx_->device.waitIdle();
}

void App::drawGui()
{
	auto &io = ImGui::GetIO();
	ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

	if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Space, false))
		paused_ = !paused_;

	drawControls();

	if (showDemo_)
		ImGui::ShowDemoWindow(&showDemo_);
}

void App::drawControls()
{
	auto &io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360, 330), ImGuiCond_FirstUseEver);
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
	ImGui::Text("%s : %u x %u mailles (%.3g m)", c.name.c_str(), c.nx, c.ny, c.dx());
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
	cb.reset();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	vkc::transitionImage(cb, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
	                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, {},
	                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite);

	vk::RenderingAttachmentInfo color{.imageView   = *swapchain_->views[imageIndex],
	                                  .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
	                                  .loadOp      = vk::AttachmentLoadOp::eClear,
	                                  .storeOp     = vk::AttachmentStoreOp::eStore,
	                                  .clearValue  = vk::ClearColorValue(kSkyBackground)};
	cb.beginRendering(vk::RenderingInfo{.renderArea           = {.offset = {0, 0}, .extent = swapchain_->extent},
	                                    .layerCount           = 1,
	                                    .colorAttachmentCount = 1,
	                                    .pColorAttachments    = &color});
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cb);
	cb.endRendering();

	vkc::transitionImage(cb, image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
	                     vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
	                     vk::PipelineStageFlagBits2::eBottomOfPipe, {});
	cb.end();
}

void App::onSwapchainRecreated()
{
	ImGui_ImplVulkan_SetMinImageCount(swapchain_->minImageCount);
	if (swapchain_->surfaceFormat.format != colorFormat_)
		throw std::runtime_error("swapchain format changed on recreation (unsupported)");
}

}        // namespace swe
