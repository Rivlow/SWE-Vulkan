#include "solver/Solver.h"

#include "util/Log.h"
#include "vk/Context.h"
#include "vk/Pipelines.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace swe {

namespace {

constexpr uint32_t kTile = 16;        // [numthreads(16, 16, 1)]
constexpr uint32_t kLine = 256;       // [numthreads(256, 1, 1)]

constexpr char const *kEntries[] = {"reduce_smax",  "finalize_dt", "flux_x",          "flux_y",       "source",
                                    "bc_inflow_norm", "bc_fluxes", "residual",        "combine_fix_dry", "final_update",
                                    "diag_partial", "diag_final",  "probes"};

// binding -> descriptor type (see shaders/solver/params.slang)
constexpr uint32_t kBindings = 15;

vk::DescriptorType bindingType(uint32_t b)
{
	return b == 0 ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer;
}

}        // namespace

Solver::Solver(vkc::Context const &ctx) :
    ctx_(ctx)
{
	createLayout();
	createPipelines();

	auto cbs = vk::raii::CommandBuffers(ctx_.device, {.commandPool        = *ctx_.commandPool,
	                                                  .level              = vk::CommandBufferLevel::ePrimary,
	                                                  .commandBufferCount = 1 + kReadbackSlots});
	stepCmd_ = std::move(cbs[0]);
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		readbackCmd_[s] = std::move(cbs[1 + s]);

	paramsBuf_ = vkc::Buffer::host(ctx_, sizeof(SolverParams), vk::BufferUsageFlagBits::eUniformBuffer, "solver-params");
	probeBuf_  = vkc::Buffer::host(ctx_, (1 + MAX_PROBES) * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, "probe-cells");
	std::memset(probeBuf_.mapped(), 0, probeBuf_.size());
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		readback_[s] = vkc::Buffer::host(ctx_, S_COUNT * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst,
		                                 std::format("solver-readback-{}", s));
}

void Solver::createLayout()
{
	// Vertex/fragment stages too: the renderer binds this set read-only to fetch Q and zb.
	auto stages = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	std::array<vk::DescriptorSetLayoutBinding, kBindings> bindings;
	for (uint32_t b = 0; b < kBindings; ++b)
		bindings[b] = {.binding = b, .descriptorType = bindingType(b), .descriptorCount = 1, .stageFlags = stages};
	setLayout_ = vk::raii::DescriptorSetLayout(ctx_.device, {.bindingCount = kBindings, .pBindings = bindings.data()});

	vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(StagePC)};
	pipelineLayout_ = vk::raii::PipelineLayout(ctx_.device, {.setLayoutCount         = 1,
	                                                         .pSetLayouts            = &*setLayout_,
	                                                         .pushConstantRangeCount = 1,
	                                                         .pPushConstantRanges    = &range});

	std::array<vk::DescriptorPoolSize, 2> sizes{{{vk::DescriptorType::eUniformBuffer, 1},
	                                             {vk::DescriptorType::eStorageBuffer, kBindings - 1}}};
	pool_ = vk::raii::DescriptorPool(ctx_.device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                               .maxSets       = 1,
	                                               .poolSizeCount = static_cast<uint32_t>(sizes.size()),
	                                               .pPoolSizes    = sizes.data()});
	auto sets = vk::raii::DescriptorSets(ctx_.device, {.descriptorPool     = *pool_,
	                                                   .descriptorSetCount = 1,
	                                                   .pSetLayouts        = &*setLayout_});
	set_      = std::move(sets[0]);
}

void Solver::createPipelines()
{
	module_ = vkc::loadShader(ctx_, "solver");
	for (auto *entry : kEntries)
		pipelines_.emplace(entry, vkc::makeComputePipeline(ctx_, module_, entry, pipelineLayout_));
}

void Solver::load(CaseData const &data)
{
	ctx_.device.waitIdle();
	if (data.nx < 3 || data.ny < 3 || data.zb.size() != data.cells() || data.h0.size() != data.cells())
		throw std::runtime_error("Solver::load: inconsistent case data");
	case_       = data;
	schemeName_ = data.scheme;
	tableau_    = scheme(schemeName_);

	allocate();
	writeDescriptors();
	updateParams();
	uploadInitialState();
	recordStep();
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		recordReadback(s);
	log::info("solver: case '{}' {}x{} ({} cells), scheme {}", case_.name, case_.nx, case_.ny, case_.cells(), schemeName_);
}

void Solver::allocate()
{
	vk::DeviceSize const n     = case_.cells();
	vk::DeviceSize const fx    = vk::DeviceSize(case_.nx + 1) * case_.ny;
	vk::DeviceSize const gy    = vk::DeviceSize(case_.nx) * (case_.ny + 1);
	vk::DeviceSize const f4    = sizeof(float) * 4;
	auto const           vert  = vk::BufferUsageFlagBits::eVertexBuffer;        // renderer may fetch through it

	scalars_ = vkc::Buffer::storage(ctx_, S_COUNT * sizeof(uint32_t), "scalars");
	zb_      = vkc::Buffer::storage(ctx_, n * sizeof(float), "zb", vert);
	Q_       = vkc::Buffer::storage(ctx_, n * f4, "Q", vert);
	Q_old_   = vkc::Buffer::storage(ctx_, n * f4, "Q_old");
	K_       = vkc::Buffer::storage(ctx_, n * f4 * tableau_.stages(), "K");
	F_L_     = vkc::Buffer::storage(ctx_, fx * f4, "F_L");
	F_R_     = vkc::Buffer::storage(ctx_, fx * f4, "F_R");
	G_D_     = vkc::Buffer::storage(ctx_, gy * f4, "G_D");
	G_U_     = vkc::Buffer::storage(ctx_, gy * f4, "G_U");
	S_b_     = vkc::Buffer::storage(ctx_, n * sizeof(float) * 2, "S_b");
	sink_    = vkc::Buffer::storage(ctx_, std::max<vk::DeviceSize>(case_.sink.size(), 1) * sizeof(uint32_t), "sink");
	inflow_  = vkc::Buffer::storage(ctx_, 4 * sizeof(float), "inflow_norm");
	diag_    = vkc::Buffer::storage(ctx_, vk::DeviceSize(vkc::divUp(case_.ny, kTile)) * vkc::divUp(case_.nx, kTile) * f4, "diag");

	// Deterministic content everywhere (never-written boundary face slots included).
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		for (auto *b : {&scalars_, &zb_, &Q_, &Q_old_, &K_, &F_L_, &F_R_, &G_D_, &G_U_, &S_b_, &sink_, &inflow_, &diag_})
			cb.fillBuffer(b->handle(), 0, vk::WholeSize, 0);
	});
	setProbes({});
}

void Solver::writeDescriptors()
{
	std::array<vkc::Buffer const *, kBindings> buffers{&paramsBuf_, &scalars_, &zb_,    &Q_,     &Q_old_,
	                                                   &K_,         &F_L_,     &F_R_,   &G_D_,   &G_U_,
	                                                   &S_b_,       &sink_,    &inflow_, &probeBuf_, &diag_};
	std::array<vk::DescriptorBufferInfo, kBindings> infos;
	std::array<vk::WriteDescriptorSet, kBindings>   writes;
	for (uint32_t b = 0; b < kBindings; ++b)
	{
		infos[b]  = buffers[b]->info();
		writes[b] = {.dstSet          = *set_,
		             .dstBinding      = b,
		             .descriptorCount = 1,
		             .descriptorType  = bindingType(b),
		             .pBufferInfo     = &infos[b]};
	}
	ctx_.device.updateDescriptorSets(writes, nullptr);
}

void Solver::uploadInitialState()
{
	size_t const       n = case_.cells();
	std::vector<float> q(n * 4, 0.0f);
	for (size_t k = 0; k < n; ++k)
	{
		q[4 * k + 0] = case_.h0[k];
		q[4 * k + 1] = case_.hu0.empty() ? 0.0f : case_.hu0[k];
		q[4 * k + 2] = case_.hv0.empty() ? 0.0f : case_.hv0[k];
	}
	Q_.upload(ctx_, q.data(), q.size() * sizeof(float));
	zb_.upload(ctx_, case_.zb.data(), n * sizeof(float));
	if (!case_.sink.empty())
		sink_.upload(ctx_, case_.sink.data(), case_.sink.size() * sizeof(uint32_t));
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) { cb.fillBuffer(scalars_.handle(), 0, vk::WholeSize, 0); });
	for (auto &r : readback_)
		std::memset(r.mapped(), 0, r.size());
}

void Solver::reset()
{
	ctx_.device.waitIdle();
	uploadInitialState();
}

void Solver::updateParams()
{
	// Constants folded in double like Taichi folds the Python expressions, then rounded to f32.
	auto const  &c   = case_;
	double const dx  = c.dx(), dy = c.dy();
	double const K_v = 3.0;        // K_VENKATAKRISHNAN
	SolverParams p{};
	p.nx      = c.nx;
	p.ny      = c.ny;
	p.n_cells = static_cast<uint32_t>(c.cells());
	p.dx      = static_cast<float>(dx);
	p.dy      = static_cast<float>(dy);
	p.g       = static_cast<float>(c.g);
	p.cfl_dx  = static_cast<float>(c.cfl * std::min(dx, dy));
	p.h_eps   = static_cast<float>(c.h_eps);
	p.h_dry   = static_cast<float>(c.h_dry);
	p.h_dry4  = static_cast<float>(std::pow(c.h_dry, 4));
	p.eps2    = static_cast<float>(std::pow(K_v * std::min(dx, dy) * 10.0 / std::max(c.Lx, c.Ly), 3));
	p.g_n2    = static_cast<float>(c.g * c.manning_n * c.manning_n);
	// a (nearly) dry domain has no wave speed: cap dt as if 5 cm of water were present everywhere, so the
	// first drop of water is resolved and the clock stays meaningful (no effect on the reference cases)
	p.dt_max  = static_cast<float>(c.cfl * std::min(dx, dy) / std::sqrt(c.g * 0.05));
	p.flags   = (c.sink.empty() ? 0u : FLAG_SINK) | (c.manning_n > 0.0 ? FLAG_FRICTION : 0u);
	for (int s = 0; s < 4; ++s)
	{
		p.bc_kind[s]  = static_cast<uint32_t>(c.bc[s].kind);
		p.bc_value[s] = static_cast<float>(c.bc[s].value);
		if (c.bc[s].kind == BcKind::Inflow)
			p.flags |= FLAG_INFLOW;
	}
	params_ = p;
	// The uniform buffer may be read by in-flight steps: only write it when the device is idle.
	ctx_.device.waitIdle();
	std::memcpy(paramsBuf_.mapped(), &params_, sizeof(params_));
}

void Solver::setScheme(std::string const &name)
{
	if (name == schemeName_)
		return;
	ctx_.device.waitIdle();
	auto const &t   = scheme(name);
	bool        grow = t.stages() > tableau_.stages();
	schemeName_      = name;
	case_.scheme     = name;
	tableau_         = t;
	if (grow)
	{
		K_ = vkc::Buffer::storage(ctx_, case_.cells() * sizeof(float) * 4 * tableau_.stages(), "K");
		ctx_.oneShot([&](vk::raii::CommandBuffer &cb) { cb.fillBuffer(K_.handle(), 0, vk::WholeSize, 0); });
		writeDescriptors();
	}
	recordStep();
}

void Solver::setCfl(double cfl)
{
	case_.cfl = cfl;
	updateParams();
}

void Solver::setManning(double n)
{
	case_.manning_n = n;
	updateParams();
}

void Solver::setBoundary(int side, BoundaryCondition const &bc)
{
	case_.bc.at(side) = bc;
	updateParams();
}

void Solver::bind(vk::raii::CommandBuffer const &cb, char const *entry) const
{
	cb.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines_.at(entry));
}

void Solver::dispatchGrid(vk::raii::CommandBuffer const &cb, char const *entry) const
{
	bind(cb, entry);
	cb.dispatch(vkc::divUp(case_.ny, kTile), vkc::divUp(case_.nx, kTile), 1);
}

void Solver::pushStage(vk::raii::CommandBuffer &cb, StagePC const &pc) const
{
	cb.pushConstants<StagePC>(*pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, pc);
}

// One full time step (see the dispatch order at the top of shaders/solver/solver.slang).
void Solver::recordStep()
{
	auto &cb = stepCmd_;
	cb.reset();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});
	cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, *set_, nullptr);

	// Whatever ran before (previous step, readback copy, rendering) must be done with the state.
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eAllCommands,
	                   vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	                   vk::PipelineStageFlagBits2::eComputeShader,
	                   vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
	                       vk::AccessFlagBits2::eUniformRead);

	dispatchGrid(cb, "reduce_smax");        // s_max (+ Q -> Q_old)
	vkc::computeBarrier(cb);

	auto coefsPC = [](std::vector<double> const &coefs) {
		StagePC pc{};
		pc.n_coefs = static_cast<uint32_t>(coefs.size());
		if (coefs.size() > pc.coef.size())
			throw std::runtime_error("Runge-Kutta scheme with more than 8 stages");
		for (size_t m = 0; m < coefs.size(); ++m)
			pc.coef[m] = static_cast<float>(coefs[m]);
		return pc;
	};

	bool first = true;
	for (int s = 0; s < tableau_.stages(); ++s)
	{
		if (!tableau_.is_used(s))
			continue;
		if (s > 0)
		{
			// Q_s = Q^n + dt sum_j A[s][j] K_j   (Q_0 = Q^n: Q already holds Q^n)
			pushStage(cb, coefsPC(tableau_.A[s]));
			dispatchGrid(cb, "combine_fix_dry");
			vkc::computeBarrier(cb);
		}
		if (first)
		{
			// dt is first needed by the stage-1 combination: compute it alongside the stage-0 fluxes
			bind(cb, "finalize_dt");
			cb.dispatch(1, 1, 1);
			first = false;
		}
		dispatchGrid(cb, "flux_x");
		dispatchGrid(cb, "flux_y");
		dispatchGrid(cb, "source");
		bind(cb, "bc_inflow_norm");
		cb.dispatch(4, 1, 1);
		vkc::computeBarrier(cb);

		bind(cb, "bc_fluxes");
		cb.dispatch(vkc::divUp(std::max(case_.nx, case_.ny), kLine), 1, 1);
		vkc::computeBarrier(cb);

		StagePC pc{};
		pc.stage = static_cast<uint32_t>(s);
		pushStage(cb, pc);
		dispatchGrid(cb, "residual");
		vkc::computeBarrier(cb);
	}

	// Q^{n+1} = Q^n + dt sum_s b[s] K_s, fix_dry, friction, outlet
	pushStage(cb, coefsPC(tableau_.b));
	dispatchGrid(cb, "final_update");
	cb.end();
}

void Solver::recordReadback(uint32_t slot)
{
	auto &cb = readbackCmd_[slot];
	cb.reset();
	cb.begin({});
	cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, *set_, nullptr);

	// diagnostics (volume, wet area, maxima) and probes of the state after this frame's steps
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
	                   vk::PipelineStageFlagBits2::eComputeShader,
	                   vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
	dispatchGrid(cb, "diag_partial");
	vkc::computeBarrier(cb);
	bind(cb, "diag_final");
	cb.dispatch(1, 1, 1);
	bind(cb, "probes");
	cb.dispatch(1, 1, 1);

	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	cb.copyBuffer(scalars_.handle(), readback_[slot].handle(),
	              vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = S_COUNT * sizeof(uint32_t)});
	// Make the copy visible to the host once the submission's fence has signalled.
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	                   vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
	cb.end();
}

void Solver::appendSteps(std::vector<vk::CommandBufferSubmitInfo> &batch, uint32_t steps, uint32_t slot)
{
	for (uint32_t k = 0; k < steps; ++k)
		batch.push_back({.commandBuffer = *stepCmd_});
	batch.push_back({.commandBuffer = *readbackCmd_[slot]});
}

static SolverScalars decodeScalars(uint32_t const *s)
{
	auto          f = [s](uint32_t k) { return std::bit_cast<float>(s[k]); };
	SolverScalars r{.s_max    = f(S_SMAX),
	                .dt       = f(S_DT),
	                .t        = f(S_TIME),
	                .n_steps  = s[S_NSTEPS],
	                .volume   = f(S_VOLUME),
	                .wetCells = f(S_WET_CELLS),
	                .hMax     = f(S_HMAX),
	                .uMax     = f(S_UMAX)};
	for (uint32_t p = 0; p < MAX_PROBES; ++p)
		r.probes[p] = {f(S_PROBES + 4 * p), f(S_PROBES + 4 * p + 1), f(S_PROBES + 4 * p + 2), f(S_PROBES + 4 * p + 3)};
	return r;
}

void Solver::setProbes(std::vector<uint32_t> const &cells)
{
	// read by in-flight readbacks: only written with the device idle
	ctx_.device.waitIdle();
	auto    *p = static_cast<uint32_t *>(probeBuf_.mapped());
	uint32_t n = static_cast<uint32_t>(std::min<size_t>(cells.size(), MAX_PROBES));
	p[0]       = n;
	for (uint32_t k = 0; k < n; ++k)
		p[1 + k] = std::min<uint32_t>(cells[k], static_cast<uint32_t>(case_.cells()) - 1);
}

SolverScalars Solver::readback(uint32_t slot) const
{
	return decodeScalars(static_cast<uint32_t const *>(readback_[slot].mapped()));
}

void Solver::run(uint32_t steps)
{
	constexpr uint32_t kBatch = 256;
	while (steps > 0)
	{
		uint32_t                                 n = std::min(steps, kBatch);
		std::vector<vk::CommandBufferSubmitInfo> infos(n, vk::CommandBufferSubmitInfo{.commandBuffer = *stepCmd_});
		ctx_.queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = n, .pCommandBufferInfos = infos.data()}, nullptr);
		steps -= n;
	}
	ctx_.queue.waitIdle();
}

SolverScalars Solver::scalarsSync() const
{
	auto s = scalars_.download<uint32_t>(ctx_, S_COUNT);
	return decodeScalars(s.data());
}

std::vector<float> Solver::stateSync() const
{
	size_t const n   = case_.cells();
	auto         q4  = Q_.download<float>(ctx_, n * 4);
	std::vector<float> q3(n * 3);
	for (size_t k = 0; k < n; ++k)
		for (int c = 0; c < 3; ++c)
			q3[3 * k + c] = q4[4 * k + c];
	return q3;
}

std::vector<float> Solver::bedSync() const
{
	return zb_.download<float>(ctx_, case_.cells());
}

}        // namespace swe
