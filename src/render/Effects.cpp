#include "render/Effects.h"

#include "solver/Solver.h"
#include "vk/Context.h"
#include "vk/Pipelines.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

namespace swe {

namespace {

constexpr uint32_t kTile = 16;

constexpr char const *kEntries[] = {"compute_wet", "blur_x",  "blur_y",          "compute_lighting",
                                    "effects_begin", "advect_copy", "advect", "update_streamlines"};

// bindings 1..14 of shaders/effects/effects_data.slang
enum Field : uint32_t
{
	WET = 1, WET_TMP, WET_BLUR, SHADOW, AO, TURB, TURB_TMP, FOAM, FOAM_TMP, WET_MEM, FLOODED,
	SEEDS, LINE_PTS, STATE,
	BINDINGS        // = 15
};

}        // namespace

Effects::Effects(vkc::Context const &ctx, Solver const &solver) :
    ctx_(ctx), solver_(solver)
{
	auto stages = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	std::array<vk::DescriptorSetLayoutBinding, BINDINGS> bindings;
	for (uint32_t b = 0; b < BINDINGS; ++b)
		bindings[b] = {.binding         = b,
		               .descriptorType  = b == 0 ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer,
		               .descriptorCount = 1,
		               .stageFlags      = stages};
	setLayout_ = vk::raii::DescriptorSetLayout(ctx_.device, {.bindingCount = BINDINGS, .pBindings = bindings.data()});

	std::array<vk::DescriptorSetLayout, 2> layouts{solver_.setLayout(), *setLayout_};
	vk::PushConstantRange                  range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(EffectsPC)};
	pipelineLayout_ = vk::raii::PipelineLayout(ctx_.device, {.setLayoutCount         = 2,
	                                                         .pSetLayouts            = layouts.data(),
	                                                         .pushConstantRangeCount = 1,
	                                                         .pPushConstantRanges    = &range});

	std::array<vk::DescriptorPoolSize, 2> sizes{{{vk::DescriptorType::eUniformBuffer, 1},
	                                             {vk::DescriptorType::eStorageBuffer, BINDINGS - 1}}};
	pool_ = vk::raii::DescriptorPool(ctx_.device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                               .maxSets       = 1,
	                                               .poolSizeCount = static_cast<uint32_t>(sizes.size()),
	                                               .pPoolSizes    = sizes.data()});
	auto sets = vk::raii::DescriptorSets(ctx_.device, {.descriptorPool = *pool_, .descriptorSetCount = 1, .pSetLayouts = &*setLayout_});
	set_      = std::move(sets[0]);

	module_ = vkc::loadShader(ctx_, "effects");
	for (auto *e : kEntries)
		pipelines_.emplace(e, vkc::makeComputePipeline(ctx_, module_, e, pipelineLayout_));

	paramsBuf_ = vkc::Buffer::host(ctx_, sizeof(EffectsParams), vk::BufferUsageFlagBits::eUniformBuffer, "effects-params");
}

void Effects::setCase(CaseData const &c)
{
	ctx_.device.waitIdle();
	nx_ = c.nx;
	ny_ = c.ny;
	double L     = std::min(c.dx(), c.dy());
	auto   p     = EffectsParams{};
	p.shore_radius = static_cast<uint32_t>(std::clamp(std::lround(12.0 / L), 1L, 6L));        // ~12 m of shore
	p.flags        = params_.flags;
	p.L_ref        = static_cast<float>(L);
	p.foam_length  = static_cast<float>(25.0 * L);
	p.turb_length  = static_cast<float>(400.0 * L);
	p.settle_time  = static_cast<float>(500.0 * L);
	p.dry_time     = static_cast<float>(2000.0 * L);
	p.ds           = static_cast<float>(1.6 * L);
	p.z_top        = *std::max_element(c.zb.begin(), c.zb.end());
	p.u_min_lines  = 0.15f;
	p.Lx           = static_cast<float>(c.Lx);
	p.Ly           = static_cast<float>(c.Ly);
	params_        = p;
	writeParams();

	vk::DeviceSize n = c.cells() * sizeof(float);
	fields_.clear();
	fields_.resize(BINDINGS);
	static char const *names[BINDINGS] = {"", "wet", "wet_tmp", "wet_blur", "shadow", "ao", "turb", "turb_tmp", "foam",
	                                      "foam_tmp", "wet_mem", "flooded", "seeds", "line_pts", "fx_state"};
	for (uint32_t b = WET; b <= FLOODED; ++b)
		fields_[b] = vkc::Buffer::storage(ctx_, n, names[b]);
	fields_[SEEDS]    = vkc::Buffer::storage(ctx_, kLines * 16, names[SEEDS]);
	fields_[LINE_PTS] = vkc::Buffer::storage(ctx_, kLines * kPoints * 16, names[LINE_PTS]);
	fields_[STATE]    = vkc::Buffer::storage(ctx_, 16, names[STATE]);

	std::array<vk::DescriptorBufferInfo, BINDINGS> infos;
	std::array<vk::WriteDescriptorSet, BINDINGS>   writes;
	for (uint32_t b = 0; b < BINDINGS; ++b)
	{
		infos[b]  = b == 0 ? paramsBuf_.info() : fields_[b].info();
		writes[b] = {.dstSet          = *set_,
		             .dstBinding      = b,
		             .descriptorCount = 1,
		             .descriptorType  = b == 0 ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer,
		             .pBufferInfo     = &infos[b]};
	}
	ctx_.device.updateDescriptorSets(writes, nullptr);

	litZScale_ = -1.0f;        // lighting recomputed on the next frame
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		uint32_t one = std::bit_cast<uint32_t>(1.0f);
		cb.fillBuffer(fields_[SHADOW].handle(), 0, vk::WholeSize, one);
		cb.fillBuffer(fields_[AO].handle(), 0, vk::WholeSize, one);
	});
	reset();
}

void Effects::reset()
{
	ctx_.device.waitIdle();
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		for (uint32_t b : {WET, WET_TMP, WET_BLUR, TURB, TURB_TMP, FOAM, FOAM_TMP, WET_MEM, FLOODED, SEEDS, STATE})
			cb.fillBuffer(fields_[b].handle(), 0, vk::WholeSize, 0);
		// dead streamline points until the first update
		cb.fillBuffer(fields_[LINE_PTS].handle(), 0, vk::WholeSize, std::bit_cast<uint32_t>(-1.0f));
	});
	resetLines_ = true;
}

void Effects::setTurbidity(bool on)
{
	if (on == turbidity())
		return;
	params_.flags = on ? (params_.flags | EFFECTS_TURBIDITY) : (params_.flags & ~EFFECTS_TURBIDITY);
	writeParams();
}

void Effects::setGroundwaterLines(bool on)
{
	if (on == groundwaterLines())
		return;
	params_.flags = on ? (params_.flags | EFFECTS_GW_LINES) : (params_.flags & ~EFFECTS_GW_LINES);
	writeParams();
	resetLines_ = true;
}

void Effects::writeParams()
{
	// read by in-flight frames: only written with the device idle
	ctx_.device.waitIdle();
	std::memcpy(paramsBuf_.mapped(), &params_, sizeof(params_));
}

uint32_t Effects::substeps(double tFrame, double sMax) const
{
	// do not jump more than ~1.5 cells at a time (update_effects)
	double n = std::ceil(tFrame * sMax / (1.5 * params_.L_ref));
	if (!std::isfinite(n))
		return 1;
	return static_cast<uint32_t>(std::clamp(n, 1.0, 24.0));
}

void Effects::bind(vk::raii::CommandBuffer const &cb, char const *entry) const
{
	cb.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines_.at(entry));
}

void Effects::dispatchGrid(vk::raii::CommandBuffer const &cb, char const *entry) const
{
	bind(cb, entry);
	cb.dispatch(vkc::divUp(ny_, kTile), vkc::divUp(nx_, kTile), 1);
}

void Effects::push(vk::raii::CommandBuffer const &cb, EffectsPC const &pc) const
{
	cb.pushConstants<EffectsPC>(*pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, pc);
}

void Effects::record(vk::raii::CommandBuffer const &cb, EffectsFrame const &f)
{
	std::array<vk::DescriptorSet, 2> sets{solver_.set(), *set_};
	cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, sets, nullptr);

	// solver steps (compute) and the previous frame's rendering (vertex/fragment reads) are done
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eAllCommands,
	                   vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	                   vk::PipelineStageFlagBits2::eComputeShader,
	                   vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
	                       vk::AccessFlagBits2::eUniformRead);

	EffectsPC pc{.z_scale = f.zScale, .frame_dt = f.realDt, .n_sub = std::max(f.substeps, 1u), .seed = ++frameCounter_};

	// relief lighting: only when the vertical exaggeration changes
	if (f.zScale != litZScale_)
	{
		push(cb, pc);
		dispatchGrid(cb, "compute_lighting");
		litZScale_ = f.zScale;
	}

	// shore mask + frame time (independent)
	push(cb, pc);
	bind(cb, "effects_begin");
	cb.dispatch(1, 1, 1);
	dispatchGrid(cb, "compute_wet");
	vkc::computeBarrier(cb);
	dispatchGrid(cb, "blur_x");
	vkc::computeBarrier(cb);
	dispatchGrid(cb, "blur_y");

	// effects carried by the flow, in substeps
	for (uint32_t k = 0; k < pc.n_sub; ++k)
	{
		pc.sub = k;
		push(cb, pc);
		vkc::computeBarrier(cb);
		dispatchGrid(cb, "advect_copy");
		vkc::computeBarrier(cb);
		dispatchGrid(cb, "advect");
	}

	if (f.streamlines || resetLines_)
	{
		pc.reset   = resetLines_ ? 1u : 0u;
		resetLines_ = false;
		push(cb, pc);
		vkc::computeBarrier(cb);
		bind(cb, "update_streamlines");
		cb.dispatch(vkc::divUp(kLines, 64), 1, 1);
	}
	// the renderer's barrier (compute -> vertex/fragment) makes all of this visible to the draws
}

}        // namespace swe
