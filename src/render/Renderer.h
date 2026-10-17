#pragma once
#include "cases/Case.h"
#include "render/Camera.h"
#include "render/SceneParams.h"
#include "vk/Buffer.h"
#include "vk/Image.h"
#include "vk/Texture.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>

namespace swe {

namespace vkc { class Context; }
class Solver;
class Effects;

struct RenderSettings
{
	float     z_scale   = 2.0f;
	bool      showWater = true;
	bool      showMesh  = false;
	bool      showLines = true;
	ColorMode colorMode = ColorMode::Realistic;
	float     cmapMin = 0.0f, cmapMax = 1.0f, cmapOpacity = 0.9f;
	bool      showOrtho = true;        // when the case has an orthophoto
	float     orthoMix  = 1.0f;
	bool      thinWater = true;        // minimum opacity for thin films
	bool      showZones = true;        // tint solver zones
	// selected cell rectangle (inclusive), none when i1 < i0
	glm::uvec4 selection{1, 0, 0, 0};
	// 3D view rectangle in pixels (the central area left free by the docked panels); empty = whole target
	glm::vec4 viewport{0.0f};
};

// 3D view of the simulation: bed, water and model base, drawn straight from the solver's buffers
// (Q and zb are read in the vertex shaders through the solver descriptor set).
class Renderer
{
  public:
	static constexpr uint32_t kFrames = 2;

	Renderer(vkc::Context const &ctx, Solver const &solver, Effects const &effects, vk::Format colorFormat);

	// Rebuilds the per-case resources (index buffer, base depth...). Call after Solver::load.
	void setCase(CaseData const &data);
	// Orthophoto covering the whole domain (row 0 = north), or clears it (rgba = nullptr). Waits for the device.
	void setOrthophoto(uint32_t width, uint32_t height, uint8_t const *rgba);
	bool hasOrthophoto() const { return hasOrtho_; }
	// (Re)creates the multisampled colour and depth targets.
	void resize(vk::Extent2D extent);

	// Recommended home view for the loaded case (ui/app.py: south of the domain, looking north)
	void homeCamera(OrbitCamera &camera, float zScale) const;
	float S() const { return S_; }

	// Records the scene into the frame's command buffer and resolves it into `target`, which is left in
	// COLOR_ATTACHMENT_OPTIMAL (the GUI is drawn on top afterwards).
	void record(vk::raii::CommandBuffer const &cb, uint32_t frame, vk::Image target, vk::ImageView targetView,
	            OrbitCamera const &camera, RenderSettings const &settings, float time);

	vk::SampleCountFlagBits samples() const { return samples_; }

  private:
	void createPipelines();

	vkc::Context const &ctx_;
	Solver const       &solver_;
	Effects const      &effects_;
	vk::Format          colorFormat_;
	vk::Format          depthFormat_ = vk::Format::eD32Sfloat;
	vk::SampleCountFlagBits samples_ = vk::SampleCountFlagBits::e4;
	vk::Extent2D        extent_{};

	vk::raii::DescriptorSetLayout setLayout_      = nullptr;
	vk::raii::PipelineLayout      pipelineLayout_ = nullptr;
	vk::raii::DescriptorPool      pool_           = nullptr;
	std::array<vk::raii::DescriptorSet, kFrames> sets_{nullptr, nullptr};
	std::array<vkc::Buffer, kFrames>             sceneUbo_;

	vk::raii::ShaderModule terrainModule_ = nullptr, waterModule_ = nullptr, skirtModule_ = nullptr, linesModule_ = nullptr;
	vk::raii::Pipeline     terrain_ = nullptr, terrainWire_ = nullptr, water_ = nullptr, waterWire_ = nullptr;
	vk::raii::Pipeline     skirtBed_ = nullptr, skirtWater_ = nullptr, lines_ = nullptr;

	vkc::Image   color_, depth_;
	vkc::Buffer  indices_, outside_;
	vkc::Texture ortho_, white_;
	bool         hasOrtho_ = false;
	void         writeOrthoDescriptors();
	uint32_t    indexCount_ = 0, skirtVertexCount_ = 0;

	// case-dependent scene constants
	float    S_ = 1, depthRef_ = 1, zBase_ = 0, eta0_ = 1, Lx_ = 1, Ly_ = 1;
	uint32_t flags_ = 0;
};

}        // namespace swe
