#include "app/App.h"

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
}

App::~App()
{
	if (ctx_)
		ctx_->device.waitIdle();
	shutdownImGui();
	// Vulkan objects must die before the window/GLFW.
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

	ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(340, 180), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Contrôles"))
	{
		ImGui::TextUnformatted(ctx_->properties.deviceName.data());
		ImGui::Text("%.1f fps  (%.2f ms)", io.Framerate, 1000.0f / io.Framerate);
		ImGui::Text("swapchain %ux%u, %u images", swapchain_->extent.width, swapchain_->extent.height, swapchain_->imageCount());
		ImGui::Separator();
		ImGui::TextDisabled("Solveur : jalon M1");
		ImGui::Checkbox("Démo ImGui", &showDemo_);
	}
	ImGui::End();

	if (showDemo_)
		ImGui::ShowDemoWindow(&showDemo_);
}

void App::drawFrame()
{
	auto &frame = frames_[frameIndex_];
	if (ctx_->device.waitForFences(*frame.inFlight, vk::True, UINT64_MAX) != vk::Result::eSuccess)
		throw std::runtime_error("waitForFences failed");

	auto [acquire, imageIndex] = swapchain_->swapchain.acquireNextImage(UINT64_MAX, *frame.imageAvailable, nullptr);
	if (acquire == vk::Result::eErrorOutOfDateKHR)
	{
		swapchain_->recreate();
		onSwapchainRecreated();
		return;
	}
	ctx_->device.resetFences(*frame.inFlight);

	recordFrame(frame, imageIndex);

	vk::SemaphoreSubmitInfo     waitInfo{.semaphore = *frame.imageAvailable, .stageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput};
	vk::SemaphoreSubmitInfo     signalInfo{.semaphore = *swapchain_->renderFinished[imageIndex], .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
	vk::CommandBufferSubmitInfo cbInfo{.commandBuffer = *frame.cmd};
	ctx_->queue.submit2(vk::SubmitInfo2{.waitSemaphoreInfoCount   = 1,
	                                    .pWaitSemaphoreInfos      = &waitInfo,
	                                    .commandBufferInfoCount   = 1,
	                                    .pCommandBufferInfos      = &cbInfo,
	                                    .signalSemaphoreInfoCount = 1,
	                                    .pSignalSemaphoreInfos    = &signalInfo},
	                    *frame.inFlight);

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
