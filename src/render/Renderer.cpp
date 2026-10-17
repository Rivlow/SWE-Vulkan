#include "render/Renderer.h"

#include "render/Effects.h"
#include "solver/Solver.h"
#include "util/Log.h"
#include "vk/Context.h"
#include "vk/Pipelines.h"
#include "vk/Swapchain.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace swe {

namespace {

// Sky background of the Python viewer (canvas.set_background_color)
constexpr std::array<float, 4> kBackground{0.80f, 0.87f, 0.93f, 1.0f};

glm::vec3 sunDirection()
{
	return glm::normalize(glm::vec3(-0.55f, 0.6f, -0.45f));        // sun at ~40 deg: marked relief
}

}        // namespace

Renderer::Renderer(vkc::Context const &ctx, Solver const &solver, Effects const &effects, vk::Format colorFormat) :
    ctx_(ctx), solver_(solver), effects_(effects), colorFormat_(colorFormat)
{
	// 4x MSAA if colour and depth both support it
	auto limits = ctx_.properties.limits;
	auto counts = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
	samples_    = (counts & vk::SampleCountFlagBits::e4) ? vk::SampleCountFlagBits::e4 : vk::SampleCountFlagBits::e1;

	auto const graphics = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings{
	    vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = graphics},
	    vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = graphics},
	    vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = graphics}};
	setLayout_ = vk::raii::DescriptorSetLayout(ctx_.device, {.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

	// set 0 = solver (Q, zb), set 1 = effect fields, set 2 = scene
	std::array<vk::DescriptorSetLayout, 3> layouts{solver_.setLayout(), effects_.setLayout(), *setLayout_};
	pipelineLayout_ = vk::raii::PipelineLayout(ctx_.device, {.setLayoutCount = static_cast<uint32_t>(layouts.size()),
	                                                         .pSetLayouts    = layouts.data()});

	std::array<vk::DescriptorPoolSize, 3> poolSizes{{{vk::DescriptorType::eUniformBuffer, kFrames},
	                                                 {vk::DescriptorType::eStorageBuffer, kFrames},
	                                                 {vk::DescriptorType::eCombinedImageSampler, kFrames}}};
	pool_ = vk::raii::DescriptorPool(ctx_.device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                               .maxSets       = kFrames,
	                                               .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
	                                               .pPoolSizes    = poolSizes.data()});
	std::array<vk::DescriptorSetLayout, kFrames> setLayouts;
	setLayouts.fill(*setLayout_);
	auto sets = vk::raii::DescriptorSets(ctx_.device, {.descriptorPool     = *pool_,
	                                                   .descriptorSetCount = kFrames,
	                                                   .pSetLayouts        = setLayouts.data()});
	for (uint32_t f = 0; f < kFrames; ++f)
	{
		sets_[f]     = std::move(sets[f]);
		sceneUbo_[f] = vkc::Buffer::host(ctx_, sizeof(SceneParams), vk::BufferUsageFlagBits::eUniformBuffer, std::format("scene-ubo-{}", f));
		auto info    = sceneUbo_[f].info();
		ctx_.device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet          = *sets_[f],
		                                                        .dstBinding      = 0,
		                                                        .descriptorCount = 1,
		                                                        .descriptorType  = vk::DescriptorType::eUniformBuffer,
		                                                        .pBufferInfo     = &info},
		                                nullptr);
	}

	white_ = vkc::Texture::white(ctx_);
	writeOrthoDescriptors();

	terrainModule_ = vkc::loadShader(ctx_, "terrain");
	waterModule_   = vkc::loadShader(ctx_, "water");
	skirtModule_   = vkc::loadShader(ctx_, "skirt");
	linesModule_   = vkc::loadShader(ctx_, "streamlines");
	createPipelines();
}

void Renderer::createPipelines()
{
	vkc::GraphicsPipelineDesc d{.layout      = *pipelineLayout_,
	                            .colorFormat = colorFormat_,
	                            .depthFormat = depthFormat_,
	                            .samples     = samples_};

	d.vertexModule   = *terrainModule_;
	d.vertexEntry    = "terrainVert";
	d.fragmentModule = *terrainModule_;
	d.fragmentEntry  = "terrainFrag";
	d.name           = "terrain";
	terrain_         = vkc::makeGraphicsPipeline(ctx_, d);

	// wireframe overlays: drawn on top of the filled mesh, slightly pulled towards the camera
	d.vertexEntry   = "terrainWireVert";
	d.fragmentEntry = "wireFrag";
	d.polygonMode   = vk::PolygonMode::eLine;
	d.depthWrite    = false;
	d.depthBias     = -1.0f;
	d.name          = "terrain-wire";
	terrainWire_    = vkc::makeGraphicsPipeline(ctx_, d);
	d.vertexModule  = *waterModule_;
	d.vertexEntry   = "waterWireVert";
	d.name          = "water-wire";
	waterWire_      = vkc::makeGraphicsPipeline(ctx_, d);

	d.polygonMode    = vk::PolygonMode::eFill;
	d.depthBias      = 0.0f;
	d.depthWrite     = true;
	d.vertexEntry    = "waterVert";
	d.fragmentModule = *waterModule_;
	d.fragmentEntry  = "waterFrag";
	d.blend          = vkc::Blend::DualSourceMul;
	d.name           = "water";
	water_           = vkc::makeGraphicsPipeline(ctx_, d);

	d.blend          = vkc::Blend::None;
	d.vertexModule   = *skirtModule_;
	d.vertexEntry    = "skirtBedVert";
	d.fragmentModule = *skirtModule_;
	d.fragmentEntry  = "skirtFrag";
	d.name           = "skirt-bed";
	skirtBed_        = vkc::makeGraphicsPipeline(ctx_, d);
	d.vertexEntry    = "skirtWaterVert";
	d.name           = "skirt-water";
	skirtWater_      = vkc::makeGraphicsPipeline(ctx_, d);

	// streamlines: screen-space quads blended over the water, no depth write
	d.vertexModule   = *linesModule_;
	d.vertexEntry    = "lineVert";
	d.fragmentModule = *linesModule_;
	d.fragmentEntry  = "lineFrag";
	d.blend          = vkc::Blend::Alpha;
	d.depthWrite     = false;
	d.name           = "streamlines";
	lines_           = vkc::makeGraphicsPipeline(ctx_, d);
}

void Renderer::writeOrthoDescriptors()
{
	auto const &tex = hasOrtho_ ? ortho_ : white_;
	vk::DescriptorImageInfo info{.sampler = tex.sampler(), .imageView = tex.view(), .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
	for (uint32_t f = 0; f < kFrames; ++f)
		ctx_.device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet          = *sets_[f],
		                                                        .dstBinding      = 2,
		                                                        .descriptorCount = 1,
		                                                        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
		                                                        .pImageInfo      = &info},
		                                nullptr);
}

void Renderer::setOrthophoto(uint32_t width, uint32_t height, uint8_t const *rgba)
{
	ctx_.device.waitIdle();
	hasOrtho_ = rgba != nullptr && width > 0 && height > 0;
	ortho_    = hasOrtho_ ? vkc::Texture(ctx_, width, height, rgba, true, "orthophoto") : vkc::Texture{};
	writeOrthoDescriptors();
}

void Renderer::setCase(CaseData const &c)
{
	ctx_.device.waitIdle();
	// a new case has no orthophoto until the application provides one
	hasOrtho_ = false;
	ortho_    = {};
	writeOrthoDescriptors();
	S_        = static_cast<float>(10.0 / std::max(c.Lx, c.Ly));
	depthRef_ = static_cast<float>(c.depth_ref);
	eta0_     = static_cast<float>(c.eta_0);
	Lx_       = static_cast<float>(c.Lx);
	Ly_       = static_cast<float>(c.Ly);
	zBase_    = *std::min_element(c.zb.begin(), c.zb.end()) - 0.35f * depthRef_;
	flags_    = std::max(c.Lx, c.Ly) > 2000.0 ? SCENE_LANDCOVER : 0u;

	// two triangles per quad of cell centres: (v00, v10, v11) and (v00, v11, v01), as build_indices()
	std::vector<uint32_t> idx;
	idx.reserve(size_t(c.nx - 1) * (c.ny - 1) * 6);
	for (uint32_t i = 0; i + 1 < c.nx; ++i)
		for (uint32_t j = 0; j + 1 < c.ny; ++j)
		{
			uint32_t v00 = i * c.ny + j, v10 = (i + 1) * c.ny + j, v01 = i * c.ny + j + 1, v11 = (i + 1) * c.ny + j + 1;
			idx.insert(idx.end(), {v00, v10, v11, v00, v11, v01});
		}
	// outside mask (1 dummy element when the case has none)
	std::vector<uint32_t> out(std::max<size_t>(c.outside.size(), 1), 0u);
	for (size_t k = 0; k < c.outside.size(); ++k)
		out[k] = c.outside[k];
	if (!c.outside.empty())
		flags_ |= SCENE_OUTSIDE;
	outside_ = vkc::Buffer::storage(ctx_, out.size() * sizeof(uint32_t), "outside");
	outside_.upload(ctx_, out.data(), out.size() * sizeof(uint32_t));
	for (uint32_t f = 0; f < kFrames; ++f)
	{
		auto info = outside_.info();
		ctx_.device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet          = *sets_[f],
		                                                        .dstBinding      = 1,
		                                                        .descriptorCount = 1,
		                                                        .descriptorType  = vk::DescriptorType::eStorageBuffer,
		                                                        .pBufferInfo     = &info},
		                                nullptr);
	}

	indexCount_       = static_cast<uint32_t>(idx.size());
	skirtVertexCount_ = 2 * (c.nx + c.ny) * 6;
	indices_          = vkc::Buffer(ctx_, idx.size() * sizeof(uint32_t),
	                                vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
	                                vk::MemoryPropertyFlagBits::eDeviceLocal, "grid-indices");
	indices_.upload(ctx_, idx.data(), idx.size() * sizeof(uint32_t));
}

void Renderer::resize(vk::Extent2D extent)
{
	ctx_.device.waitIdle();
	extent_ = extent;
	color_  = vkc::Image(ctx_, extent, colorFormat_,
	                     vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment, samples_, "msaa-color");
	depth_  = vkc::Image(ctx_, extent, depthFormat_,
	                     vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment, samples_, "depth");
}

void Renderer::homeCamera(OrbitCamera &camera, float zScale) const
{
	float focalY = 0.5f * eta0_ * zScale * S_;
	float xc = 0.5f * Lx_ * S_, zc = -0.5f * Ly_ * S_;        // domain centre (Z = -y)
	camera.setHome({xc, 9.0f, zc + 11.0f}, {xc, focalY, zc});
}

void Renderer::record(vk::raii::CommandBuffer const &cb, uint32_t frame, vk::Image target, vk::ImageView targetView,
                      OrbitCamera const &camera, RenderSettings const &settings, float time)
{
	// 3D view rectangle (clamped to the target)
	glm::vec4 vp = settings.viewport;
	if (vp.z < 1.0f || vp.w < 1.0f)
		vp = glm::vec4(0.0f, 0.0f, float(extent_.width), float(extent_.height));
	vp.x = std::clamp(vp.x, 0.0f, float(extent_.width) - 1.0f);
	vp.y = std::clamp(vp.y, 0.0f, float(extent_.height) - 1.0f);
	vp.z = std::clamp(vp.z, 1.0f, float(extent_.width) - vp.x);
	vp.w = std::clamp(vp.w, 1.0f, float(extent_.height) - vp.y);

	// scene constants (this frame slot is not in use by the GPU: its fence has been waited for)
	SceneParams sp;
	float       aspect = vp.z / vp.w;
	sp.view_proj       = camera.proj(aspect) * camera.view();
	sp.cam_pos         = glm::vec4(camera.position, 1.0f);
	sp.sun             = glm::vec4(sunDirection(), 0.0f);
	sp.S               = S_;
	sp.z_scale         = settings.z_scale;
	sp.depth_ref       = depthRef_;
	sp.water_clarity   = 0.6f * depthRef_;
	sp.z_base          = zBase_;
	sp.time            = time;
	sp.Lx              = Lx_;
	sp.Ly              = Ly_;
	sp.flags           = flags_ | (hasOrtho_ && settings.showOrtho ? SCENE_ORTHO : 0u);
	sp.ortho_mix       = settings.orthoMix;
	sp.color_mode      = static_cast<uint32_t>(settings.colorMode);
	sp.cmap_min        = settings.cmapMin;
	sp.cmap_max        = settings.cmapMax;
	sp.cmap_opacity    = settings.cmapOpacity;
	sp.viewport_w      = vp.z;
	sp.viewport_h      = vp.w;
	std::memcpy(sceneUbo_[frame].mapped(), &sp, sizeof(sp));

	// the solver's last step wrote Q: make it visible to the vertex/fragment fetches
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	                   vk::PipelineStageFlagBits2::eVertexShader | vk::PipelineStageFlagBits2::eFragmentShader,
	                   vk::AccessFlagBits2::eShaderStorageRead);

	auto const attachStages = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
	auto const depthStages  = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
	vkc::transitionImage(cb, target, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, attachStages, {},
	                     attachStages, vk::AccessFlagBits2::eColorAttachmentWrite);
	// targets shared by the frames in flight: previous frame's writes -> this frame's writes
	vkc::transitionImage(cb, color_.image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
	                     attachStages, vk::AccessFlagBits2::eColorAttachmentWrite, attachStages,
	                     vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead);
	vkc::transitionImage(cb, depth_.image(), vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal,
	                     depthStages, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, depthStages,
	                     vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	                     vk::ImageAspectFlagBits::eDepth);

	bool                        msaa = samples_ != vk::SampleCountFlagBits::e1;
	vk::RenderingAttachmentInfo color{.imageView          = msaa ? color_.view() : targetView,
	                                  .imageLayout        = vk::ImageLayout::eColorAttachmentOptimal,
	                                  .resolveMode        = msaa ? vk::ResolveModeFlagBits::eAverage : vk::ResolveModeFlagBits::eNone,
	                                  .resolveImageView   = msaa ? targetView : vk::ImageView{},
	                                  .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
	                                  .loadOp             = vk::AttachmentLoadOp::eClear,
	                                  .storeOp            = msaa ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore,
	                                  .clearValue         = vk::ClearColorValue(kBackground)};
	vk::RenderingAttachmentInfo depth{.imageView   = depth_.view(),
	                                  .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
	                                  .loadOp      = vk::AttachmentLoadOp::eClear,
	                                  .storeOp     = vk::AttachmentStoreOp::eDontCare,
	                                  .clearValue  = vk::ClearDepthStencilValue(1.0f, 0)};
	cb.beginRendering(vk::RenderingInfo{.renderArea           = {.offset = {0, 0}, .extent = extent_},
	                                    .layerCount           = 1,
	                                    .colorAttachmentCount = 1,
	                                    .pColorAttachments    = &color,
	                                    .pDepthAttachment     = &depth});
	cb.setViewport(0, vk::Viewport(vp.x, vp.y, vp.z, vp.w, 0.0f, 1.0f));
	cb.setScissor(0, vk::Rect2D({int32_t(vp.x), int32_t(vp.y)}, {uint32_t(vp.z), uint32_t(vp.w)}));
	std::array<vk::DescriptorSet, 3> sets{solver_.set(), effects_.set(), *sets_[frame]};
	cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout_, 0, sets, nullptr);
	cb.bindIndexBuffer(indices_.handle(), 0, vk::IndexType::eUint32);

	cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *terrain_);
	cb.drawIndexed(indexCount_, 1, 0, 0, 0);
	cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *skirtBed_);
	cb.draw(skirtVertexCount_, 1, 0, 0);
	if (settings.showMesh)
	{
		cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *terrainWire_);
		cb.drawIndexed(indexCount_, 1, 0, 0, 0);
	}
	if (settings.showWater)
	{
		cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *skirtWater_);
		cb.draw(skirtVertexCount_, 1, 0, 0);
		cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *water_);
		cb.drawIndexed(indexCount_, 1, 0, 0, 0);
		if (settings.showMesh)
		{
			cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *waterWire_);
			cb.drawIndexed(indexCount_, 1, 0, 0, 0);
		}
		if (settings.showLines)
		{
			cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *lines_);
			cb.draw(effects_.lineVertexCount(), 1, 0, 0);
		}
	}
	cb.endRendering();
}

}        // namespace swe
