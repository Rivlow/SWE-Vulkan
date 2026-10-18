#pragma once
// Visual effects computed on the GPU each frame (port of ui/renderer.py): shore mask, relief lighting,
// foam / turbidity / wet memory / flood extent carried by the flow, streamlines.
// Host mirror of shaders/effects/effects_data.slang - keep both in sync.
#include "cases/Case.h"
#include "vk/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

#include <map>
#include <string>

namespace swe {

namespace vkc { class Context; }
class Solver;

struct EffectsParams        // std140 uniform buffer
{
	uint32_t shore_radius = 1, flags = 0, _pad0 = 0, _pad1 = 0;
	float    L_ref = 1, foam_length = 1, turb_length = 1, settle_time = 1;
	float    dry_time = 1, ds = 1, z_top = 0, u_min_lines = 0.15f;
	float    Lx = 1, Ly = 1, _pad2 = 0, _pad3 = 0;
};
static_assert(sizeof(EffectsParams) == 64);

enum EffectsFlags : uint32_t
{
	EFFECTS_TURBIDITY = 1,
	EFFECTS_GW_LINES  = 2,        // streamlines of the aquifer flow
};

struct EffectsPC        // push constants
{
	float    z_scale = 1, frame_dt = 0;
	uint32_t n_sub = 1, sub = 0, seed = 0, reset = 0, _pad0 = 0, _pad1 = 0;
};
static_assert(sizeof(EffectsPC) == 32);

struct EffectsFrame
{
	float    zScale     = 1.0f;
	float    realDt     = 0.0f;        // wall-clock time of the frame (streamline lifetimes), 0 when paused
	uint32_t substeps   = 1;           // advection substeps (~1.5 cells per substep)
	bool     streamlines = true;
};

class Effects
{
  public:
	static constexpr uint32_t kLines = 110, kPoints = 24;

	Effects(vkc::Context const &ctx, Solver const &solver);

	void setCase(CaseData const &data);        // allocates the fields; waits for the device
	void reset();                              // clears the carried effects and respawns streamlines; waits
	void setTurbidity(bool on);
	bool turbidity() const { return (params_.flags & EFFECTS_TURBIDITY) != 0; }
	// streamlines of the aquifer (pore velocity -kh grad H / Sy) instead of the surface water; respawns them
	void setGroundwaterLines(bool on);
	bool groundwaterLines() const { return (params_.flags & EFFECTS_GW_LINES) != 0; }

	// Number of advection substeps for a frame advancing the simulation by tFrame at speed sMax
	uint32_t substeps(double tFrame, double sMax) const;

	// Records this frame's effect kernels (after the solver steps, before rendering).
	void record(vk::raii::CommandBuffer const &cb, EffectsFrame const &frame);

	vk::DescriptorSetLayout setLayout() const { return *setLayout_; }
	vk::DescriptorSet       set() const { return *set_; }
	uint32_t                lineVertexCount() const { return kLines * (kPoints - 1) * 6; }

  private:
	void bind(vk::raii::CommandBuffer const &cb, char const *entry) const;
	void dispatchGrid(vk::raii::CommandBuffer const &cb, char const *entry) const;
	void push(vk::raii::CommandBuffer const &cb, EffectsPC const &pc) const;
	void writeParams();

	vkc::Context const &ctx_;
	Solver const       &solver_;

	vk::raii::DescriptorSetLayout             setLayout_      = nullptr;
	vk::raii::PipelineLayout                  pipelineLayout_ = nullptr;
	vk::raii::DescriptorPool                  pool_           = nullptr;
	vk::raii::DescriptorSet                   set_            = nullptr;
	vk::raii::ShaderModule                    module_         = nullptr;
	std::map<std::string, vk::raii::Pipeline> pipelines_;

	EffectsParams params_{};
	vkc::Buffer   paramsBuf_;
	std::vector<vkc::Buffer> fields_;        // bindings 1..14
	uint32_t      nx_ = 0, ny_ = 0;

	float    litZScale_     = -1.0f;        // exaggeration the shadows/AO were computed for
	bool     resetLines_    = true;
	uint32_t frameCounter_  = 0;
};

}        // namespace swe
