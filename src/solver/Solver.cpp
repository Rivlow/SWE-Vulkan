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

// entry points of solver.slang, in Solver::Kernel order
constexpr char const *kEntries[] = {"build_tiles", "reduce_smax",  "finalize_dt", "flux_x", "flux_y",   "bc_fluxes",
                                    "drain_alpha", "euler_stage",  "diag_partial", "diag_final", "probes", "set_water",
                                    "gw_alpha",    "gw_lateral",   "vertical_step", "profile"};
constexpr uint32_t kTileGroup = 64;        // build_tiles: [numthreads(64, 1, 1)]
static_assert(std::size(kEntries) == size_t(Solver::Kernel::Count));

vk::DescriptorType bindingType(uint32_t b)
{
	return b == B_PARAMS ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer;
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
	profileCells_ = vkc::Buffer::host(ctx_, (1 + MAX_PROFILE) * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer, "profile-cells");
	std::memset(profileCells_.mapped(), 0, profileCells_.size());
	profileOut_ = vkc::Buffer::storage(ctx_, MAX_PROFILE * sizeof(ProfilePoint), "profile-out");
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		profileReadback_[s] = vkc::Buffer::host(ctx_, MAX_PROFILE * sizeof(ProfilePoint), vk::BufferUsageFlagBits::eTransferDst,
		                                        std::format("profile-readback-{}", s));
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		readback_[s] = vkc::Buffer::host(ctx_, S_COUNT * sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst,
		                                 std::format("solver-readback-{}", s));
}

void Solver::createLayout()
{
	// Vertex/fragment stages too: the renderer binds this set read-only to fetch Q and zb.
	auto stages = vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
	std::array<vk::DescriptorSetLayoutBinding, B_COUNT> bindings;
	for (uint32_t b = 0; b < B_COUNT; ++b)
		bindings[b] = {.binding = b, .descriptorType = bindingType(b), .descriptorCount = 1, .stageFlags = stages};
	setLayout_ = vk::raii::DescriptorSetLayout(ctx_.device, {.bindingCount = B_COUNT, .pBindings = bindings.data()});

	vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(StagePC)};
	pipelineLayout_ = vk::raii::PipelineLayout(ctx_.device, {.setLayoutCount         = 1,
	                                                         .pSetLayouts            = &*setLayout_,
	                                                         .pushConstantRangeCount = 1,
	                                                         .pPushConstantRanges    = &range});

	std::array<vk::DescriptorPoolSize, 2> sizes{{{vk::DescriptorType::eUniformBuffer, 1},
	                                             {vk::DescriptorType::eStorageBuffer, B_COUNT - 1}}};
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
	pipelines_.reserve(std::size(kEntries));
	for (auto *entry : kEntries)
		pipelines_.push_back(vkc::makeComputePipeline(ctx_, module_, entry, pipelineLayout_));
}

void Solver::load(CaseData const &data)
{
	ctx_.device.waitIdle();
	if (data.nx < 3 || data.ny < 3 || data.zb.size() != data.cells() || data.h0.size() != data.cells())
		throw std::runtime_error("Solver::load: inconsistent case data");
	nx_     = data.nx;
	ny_     = data.ny;
	tilesX_ = vkc::divUp(nx_, kTile);
	tilesY_ = vkc::divUp(ny_, kTile);
	srcOn_     = data.infiltration.active() || data.rain.active();
	gwOn_      = data.gw.enabled;
	gwLateral_ = gwOn_ && std::ranges::any_of(data.gw.kh, [](float k) { return k > 0.0f; });
	if (gwOn_)
		for (auto const *a : {&data.gw.kv, &data.gw.kh, &data.gw.sy, &data.gw.z_sub, &data.gw.h0})
			if (a->size() != data.cells())
				throw std::runtime_error("Solver::load: inconsistent groundwater data");

	allocate();
	writeDescriptors();
	zb_.upload(ctx_, data.zb.data(), cells() * sizeof(float));
	if (gwOn_)
	{
		std::vector<float> soil(cells() * 4);
		for (size_t k = 0; k < cells(); ++k)
		{
			soil[4 * k + 0] = data.gw.kv[k];
			soil[4 * k + 1] = data.gw.kh[k];
			soil[4 * k + 2] = data.gw.sy[k];
			soil[4 * k + 3] = data.gw.z_sub[k];
		}
		soil_.upload(ctx_, soil.data(), soil.size() * sizeof(float));
	}
	if (srcOn_)
		uploadChronologies(data);
	uploadZones(data);
	schemeName_.clear();        // forces the step to be recorded
	configure(data);
	reset(data);
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		recordReadback(s);
	log::info("solver: case '{}' {}x{} ({} cells), scheme {}", data.name, nx_, ny_, cells(), schemeName_);
}

void Solver::allocate()
{
	vk::DeviceSize const n  = cells();
	vk::DeviceSize const fx = vk::DeviceSize(nx_ + 1) * ny_;
	vk::DeviceSize const gy = vk::DeviceSize(nx_) * (ny_ + 1);
	vk::DeviceSize const f4 = sizeof(float) * 4;
	auto const           vert = vk::BufferUsageFlagBits::eVertexBuffer;        // renderer may fetch through it

	scalars_  = vkc::Buffer::storage(ctx_, S_COUNT * sizeof(uint32_t), "scalars");
	zb_       = vkc::Buffer::storage(ctx_, n * sizeof(float), "zb", vert);
	Q_        = vkc::Buffer::storage(ctx_, n * f4, "Q", vert);
	Q_old_    = vkc::Buffer::storage(ctx_, n * f4, "Q_old");
	alpha_[0] = vkc::Buffer::storage(ctx_, n * sizeof(float), "alpha_a");
	alpha_[1] = vkc::Buffer::storage(ctx_, n * sizeof(float), "alpha_b");
	massFlux_ = vkc::Buffer::storage(ctx_, n * f4, "mass_flux");
	F_L_      = vkc::Buffer::storage(ctx_, fx * f4, "F_L");
	F_R_      = vkc::Buffer::storage(ctx_, fx * f4, "F_R");
	G_D_      = vkc::Buffer::storage(ctx_, gy * f4, "G_D");
	G_U_      = vkc::Buffer::storage(ctx_, gy * f4, "G_U");
	cond_     = vkc::Buffer::storage(ctx_, n * sizeof(uint32_t), "cond");
	zones_    = vkc::Buffer::storage(ctx_, MAX_ZONES * sizeof(Zone), "zones");
	inflow_   = vkc::Buffer::storage(ctx_, 4 * sizeof(float), "inflow_norm");
	vk::DeviceSize const tiles = tileCount();
	diag_       = vkc::Buffer::storage(ctx_, tiles * f4, "diag");
	tileWet_    = vkc::Buffer::storage(ctx_, tiles * sizeof(uint32_t), "tile_wet");
	tileList_   = vkc::Buffer::storage(ctx_, tiles * sizeof(uint32_t), "tile_list");
	tileSeed_   = vkc::Buffer::storage(ctx_, tiles * sizeof(uint32_t), "tile_seed");
	// (x, y, z) of the indirect dispatches, then build_tiles' running count and finished groups
	tileArgs_   = vkc::Buffer::storage(ctx_, 5 * sizeof(uint32_t), "tile_args", vk::BufferUsageFlagBits::eIndirectBuffer);
	// groundwater: a single element when off (bound but never read)
	vk::DeviceSize const ng = gwOn_ ? n : 1;
	soil_   = vkc::Buffer::storage(ctx_, ng * f4, "soil", vert);
	gw_     = vkc::Buffer::storage(ctx_, ng * f4, "gw", vert);
	gwAux_  = vkc::Buffer::storage(ctx_, ng * f4, "gw_aux");
	diagGw_ = vkc::Buffer::storage(ctx_, (gwOn_ || srcOn_ ? 2 * tiles : 1) * f4, "diag_vertical");
	// chronologies: the table buffer is sized by uploadChronologies
	vk::DeviceSize const ns = srcOn_ ? n : 1;
	srcZones_ = vkc::Buffer::storage(ctx_, ns * sizeof(uint32_t), "src_zones");
	srcCum_   = vkc::Buffer::storage(ctx_, ns * 4 * sizeof(uint32_t), "src_cum", vert);
	chrono_   = vkc::Buffer::storage(ctx_, 8 * sizeof(float), "chrono");
	srcRates_ = vkc::Buffer::storage(ctx_, sizeof(float), "src_rates");

	// Deterministic content everywhere (never-written boundary face slots included).
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		for (auto *b : {&scalars_, &zb_, &Q_, &Q_old_, &alpha_[0], &alpha_[1], &massFlux_, &F_L_, &F_R_, &G_D_, &G_U_, &cond_, &zones_,
		                &inflow_, &diag_, &tileWet_, &tileList_, &tileArgs_, &tileSeed_, &soil_, &gw_, &gwAux_, &diagGw_,
		                &srcZones_, &srcCum_, &chrono_, &srcRates_})
			cb.fillBuffer(b->handle(), 0, vk::WholeSize, 0);
	});
	std::array<uint32_t, 5> const args{0u, 1u, 1u, 0u, 0u};
	tileArgs_.upload(ctx_, args.data(), sizeof(args));
	setProbes({});
	setProfile({});
}

void Solver::writeDescriptors()
{
	std::array<std::pair<SolverBinding, vkc::Buffer const *>, B_COUNT> const buffers{{
	    {B_PARAMS, &paramsBuf_},   {B_SCALARS, &scalars_}, {B_ZB, &zb_},       {B_Q, &Q_},
	    {B_Q_OLD, &Q_old_},        {B_ALPHA_A, &alpha_[0]}, {B_F_L, &F_L_},     {B_F_R, &F_R_},
	    {B_G_D, &G_D_},            {B_G_U, &G_U_},         {B_ALPHA_B, &alpha_[1]}, {B_COND, &cond_},
	    {B_INFLOW, &inflow_},      {B_PROBE_CELLS, &probeBuf_}, {B_DIAG, &diag_}, {B_ZONES, &zones_},
	    {B_MASS_FLUX, &massFlux_}, {B_TILE_WET, &tileWet_}, {B_TILE_LIST, &tileList_},
	    {B_TILE_ARGS, &tileArgs_}, {B_TILE_SEED, &tileSeed_}, {B_SOIL, &soil_}, {B_GW, &gw_},
	    {B_GW_AUX, &gwAux_},       {B_DIAG_GW, &diagGw_},  {B_SRC_ZONES, &srcZones_}, {B_CHRONO, &chrono_},
	    {B_SRC_RATES, &srcRates_}, {B_SRC_CUM, &srcCum_}, {B_PROFILE_CELLS, &profileCells_}, {B_PROFILE_OUT, &profileOut_},
	}};
	std::array<vk::DescriptorBufferInfo, B_COUNT> infos;
	std::array<vk::WriteDescriptorSet, B_COUNT>   writes;
	for (size_t k = 0; k < buffers.size(); ++k)
	{
		auto [b, buffer] = buffers[k];
		infos[k]  = buffer->info();
		writes[k] = {.dstSet = *set_, .dstBinding = b, .descriptorCount = 1, .descriptorType = bindingType(b), .pBufferInfo = &infos[k]};
	}
	ctx_.device.updateDescriptorSets(writes, nullptr);
}

void Solver::reset(CaseData const &data)
{
	ctx_.device.waitIdle();
	size_t const n = cells();
	if (data.h0.size() != n || data.zb.size() != n)
		throw std::runtime_error("Solver::reset: case does not match the loaded grid");
	// Q = (h, hu, hv, zb): the bed rides along in w (one load per cell in the solver kernels)
	std::vector<float> q(n * 4, 0.0f);
	for (size_t k = 0; k < n; ++k)
	{
		q[4 * k + 0] = data.h0[k];
		q[4 * k + 1] = data.hu0.empty() ? 0.0f : data.hu0[k];
		q[4 * k + 2] = data.hv0.empty() ? 0.0f : data.hv0[k];
		q[4 * k + 3] = data.zb[k];
	}
	Q_.upload(ctx_, q.data(), q.size() * sizeof(float));
	if (gwOn_)
	{
		// (h_gw, exchange rate, cumulated infiltration, cumulated exfiltration)
		std::vector<float> g(n * 4, 0.0f);
		for (size_t k = 0; k < n; ++k)
			g[4 * k] = data.gw.h0[k];
		gw_.upload(ctx_, g.data(), g.size() * sizeof(float));
		// (outflow factor, lateral change, h_gw in 64-bit fixed point: units of 2^-8 m and 2^-40 m)
		std::vector<uint32_t> aux(n * 4, 0u);
		for (size_t k = 0; k < n; ++k)
		{
			double v      = std::ldexp(double(std::max(data.gw.h0[k], 0.0f)), 8);
			double whole  = std::floor(v);
			aux[4 * k + 2] = static_cast<uint32_t>(whole);
			aux[4 * k + 3] = static_cast<uint32_t>(std::ldexp(v - whole, 32));
		}
		gwAux_.upload(ctx_, aux.data(), aux.size() * sizeof(uint32_t));
	}
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		cb.fillBuffer(srcCum_.handle(), 0, vk::WholeSize, 0);
		cb.fillBuffer(scalars_.handle(), 0, vk::WholeSize, 0);
		cb.fillBuffer(tileWet_.handle(), 0, vk::WholeSize, ~0u);        // no step count: reduceSmax stamps the wet tiles
	});
	for (auto &r : readback_)
		std::memset(r.mapped(), 0, r.size());
	reduceSmax();
}

// s_max of the current state into the accumulator of the next step (the steps keep it up to date themselves)
void Solver::reduceSmax()
{
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		vkc::transferToCompute(cb);
		cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, *set_, nullptr);
		dispatchGrid(cb, Kernel::ReduceSmax);
	});
}

void Solver::uploadZones(CaseData const &data)
{
	ctx_.device.waitIdle();
	size_t const                n = cells();
	std::vector<uint32_t>       mask(n, 0);
	std::array<Zone, MAX_ZONES> table{};
	size_t                      nz = std::min<size_t>(data.zones.size(), MAX_ZONES);
	if (data.zoneMask.size() == n)
		for (size_t k = 0; k < n; ++k)
		{
			uint32_t z = data.zoneMask[k];
			if (z >= 1 && z <= nz)
			{
				mask[k] = z;
				++table[z - 1].ncells;
			}
		}
	for (size_t z = 0; z < nz; ++z)
	{
		table[z].kind  = static_cast<uint32_t>(data.zones[z].kind);
		table[z].value = static_cast<float>(data.zones[z].value);
	}
	cond_.upload(ctx_, mask.data(), n * sizeof(uint32_t));
	zones_.upload(ctx_, table.data(), sizeof(table));
	uploadTileSeeds(data);
}

// Tiles computed even when dry: cells of source / level zones (they create water) and the sides with an
// inflow or level condition (water enters through the boundary faces).
void Solver::uploadTileSeeds(CaseData const &data)
{
	std::vector<uint32_t> seed(tileCount(), allTiles_ ? 1u : 0u);
	auto                  mark = [&](uint32_t i, uint32_t j) { seed[(i / kTile) * tilesY_ + j / kTile] = 1; };
	if (data.zoneMask.size() == cells())
		for (uint32_t i = 0; i < nx_; ++i)
			for (uint32_t j = 0; j < ny_; ++j)
				if (uint32_t z = data.zoneMask[data.index(i, j)]; z >= 1 && z <= data.zones.size() && z <= MAX_ZONES)
					if (auto kind = data.zones[z - 1].kind; kind == ZoneKind::Source || kind == ZoneKind::Level)
						mark(i, j);
	for (int s = 0; s < 4; ++s)
	{
		if (data.bc[s].kind == BcKind::Wall)
			continue;
		if (s < 2)
			for (uint32_t j = 0; j < ny_; ++j)
				mark(s == 0 ? 0 : nx_ - 1, j);
		else
			for (uint32_t i = 0; i < nx_; ++i)
				mark(i, s == 2 ? 0 : ny_ - 1);
	}
	ctx_.device.waitIdle();
	tileSeed_.upload(ctx_, seed.data(), seed.size() * sizeof(uint32_t));
	for (int s = 0; s < 4; ++s)
		seededBc_[s] = data.bc[s].kind;
}

void Solver::configure(CaseData const &data)
{
	// The uniform buffer and the recorded step may be in use by in-flight submissions.
	ctx_.device.waitIdle();
	params_ = buildSolverParams(data);
	std::memcpy(paramsBuf_.mapped(), &params_, sizeof(params_));
	for (int s = 0; s < 4; ++s)
		if (data.bc[s].kind != seededBc_[s])
		{
			uploadTileSeeds(data);        // a side opened or closed
			break;
		}

	int passes = std::clamp(data.drain_passes, 0, MAX_DRAIN_PASSES);
	if (data.scheme != schemeName_ || passes != drainPasses_)
	{
		scheme_      = scheme(data.scheme);
		schemeName_  = data.scheme;
		drainPasses_ = passes;
		recordStep();
	}
}

void Solver::editWater(CellRect const &rect, WaterEdit mode, float value)
{
	CellRect r = rect;
	r.i1 = std::min(r.i1, nx_ - 1);
	r.j1 = std::min(r.j1, ny_ - 1);
	if (r.empty())
		return;
	ctx_.device.waitIdle();
	StagePC pc{};
	pc.mode = static_cast<uint32_t>(mode);
	pc.coef = {float(r.i0), float(r.j0), float(r.i1), float(r.j1), value, 0, 0, 0};
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
		cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, *set_, nullptr);
		pushStage(cb, pc);
		dispatchGrid(cb, Kernel::SetWater);
		vkc::computeBarrier(cb);
		dispatchGrid(cb, Kernel::ReduceSmax);        // the new water may be faster than anything before
	});
}

void Solver::bind(vk::raii::CommandBuffer const &cb, Kernel k) const
{
	cb.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines_[size_t(k)]);
}

void Solver::dispatchGrid(vk::raii::CommandBuffer const &cb, Kernel k) const
{
	bind(cb, k);
	cb.dispatch(vkc::divUp(ny_, kTile), vkc::divUp(nx_, kTile), 1);
}

void Solver::dispatchActive(vk::raii::CommandBuffer const &cb, Kernel k) const
{
	bind(cb, k);
	cb.dispatchIndirect(tileArgs_.handle(), 0);
}

uint32_t Solver::activeTilesSync() const
{
	return tileArgs_.download<uint32_t>(ctx_, 1)[0];
}

void Solver::pushStage(vk::raii::CommandBuffer const &cb, StagePC const &pc) const
{
	cb.pushConstants<StagePC>(*pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, pc);
}

void Solver::recordStep()
{
	auto &cb = stepCmd_;
	cb.reset();
	cb.begin({.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});
	recordStepCommands(cb, {});
	cb.end();
}

// One full time step (see the dispatch order at the top of shaders/solver/solver.slang). `mark(label)` is
// called after each phase (and its barrier): the profiler writes a timestamp there.
void Solver::recordStepCommands(vk::raii::CommandBuffer const &cb, Marker const &mark) const
{
	auto at = [&](char const *label) {
		if (mark)
			mark(label);
	};
	cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout_, 0, *set_, nullptr);

	// Whatever ran before (previous step, readback copy, rendering) must be done with the state.
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eAllCommands,
	                   vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	                   vk::PipelineStageFlagBits2::eComputeShader,
	                   vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
	                       vk::AccessFlagBits2::eUniformRead);

	// active tiles of this step (from the wet stamps of the previous one), then the indirect arguments are ready
	bind(cb, Kernel::BuildTiles);
	cb.dispatch(vkc::divUp(tileCount(), kTileGroup), 1, 1);
	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	                   vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect,
	                   vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
	                       vk::AccessFlagBits2::eIndirectCommandRead);
	at("build_tiles");

	uint32_t const passes = static_cast<uint32_t>(drainPasses_);
	int const      S      = scheme_.nstages();
	for (int s = 0; s < S; ++s)
	{
		// Q holds Q_s (Q^n on stage 0). Interior and boundary fluxes (+ bed source) of this stage, one phase:
		if (s == 0)
		{
			// dt (from the s_max left by the previous step) is first needed by the drain limiter / Euler
			// update: compute it alongside the stage-0 fluxes
			bind(cb, Kernel::FinalizeDt);
			cb.dispatch(1, 1, 1);
		}
		dispatchActive(cb, Kernel::FluxX);
		dispatchActive(cb, Kernel::FluxY);
		bind(cb, Kernel::BcFluxes);
		cb.dispatch(4, 1, 1);        // one workgroup per side
		vkc::computeBarrier(cb);
		at("fluxes (flux_x, flux_y, bc_fluxes, finalize_dt)");

		// drain limiter: K Jacobi passes, each reading the factors of the previous one
		for (uint32_t k = 0; k < passes; ++k)
		{
			StagePC pc{};
			pc.pass = k;
			pushStage(cb, pc);
			dispatchActive(cb, Kernel::DrainAlpha);
			vkc::computeBarrier(cb);
			at(k == 0 ? "drain_alpha (pass 0)" : "drain_alpha (passes 1+)");
		}

		// Q_{s+1} = a_s Q^n + b_s (Q_s + dt L(Q_s)), fix_dry; last stage: friction, zones and s_max too
		StagePC pc{};
		pc.flags   = (s == 0 ? STAGE_FIRST : 0u) | (s == S - 1 ? STAGE_LAST : 0u);
		pc.pass    = passes;
		pc.coef[0] = static_cast<float>(scheme_.stages[s].first);
		pc.coef[1] = static_cast<float>(scheme_.stages[s].second);
		pushStage(cb, pc);
		dispatchActive(cb, Kernel::EulerStage);
		if (s + 1 < S || mark || gwOn_ || srcOn_)
			vkc::computeBarrier(cb);
		at("euler_stage");
	}

	// vertical terms, full grid: groundwater lateral flow (outflow factors, then the change of h_gw), then the
	// chronologies and the exchange with the aquifer
	if (gwOn_ || srcOn_)
	{
		if (gwLateral_)
		{
			dispatchGrid(cb, Kernel::GwAlpha);
			vkc::computeBarrier(cb);
			dispatchGrid(cb, Kernel::GwLateral);
			vkc::computeBarrier(cb);
			at("groundwater lateral (gw_alpha, gw_lateral)");
		}
		dispatchGrid(cb, Kernel::Vertical);
		if (mark)
			vkc::computeBarrier(cb);
		at("vertical_step");
	}
}

std::vector<Solver::PhaseTime> Solver::profileStep(uint32_t repeats)
{
	ctx_.device.waitIdle();
	double const period = ctx_.properties.limits.timestampPeriod;        // ns per tick
	if (period <= 0.0)
		return {};
	constexpr uint32_t   kMaxMarks = 64;
	vk::raii::QueryPool  pool(ctx_.device, {.queryType = vk::QueryType::eTimestamp, .queryCount = kMaxMarks});
	auto                 cbs = vk::raii::CommandBuffers(ctx_.device, {.commandPool        = *ctx_.commandPool,
	                                                                 .level              = vk::CommandBufferLevel::ePrimary,
	                                                                 .commandBufferCount = 1});
	auto                &cb  = cbs[0];
	std::vector<std::string> labels;
	cb.begin({});
	cb.resetQueryPool(*pool, 0, kMaxMarks);
	cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool, 0);
	recordStepCommands(cb, [&](char const *label) {
		if (labels.size() + 1 < kMaxMarks)
		{
			labels.emplace_back(label);
			cb.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *pool, uint32_t(labels.size()));
		}
	});
	cb.end();

	// average over `repeats` steps (each one advances the simulation), summed per label
	std::vector<PhaseTime> phases;
	for (auto const &l : labels)
		if (std::ranges::none_of(phases, [&](PhaseTime const &p) { return p.label == l; }))
			phases.push_back({l, 0.0});
	uint32_t n = static_cast<uint32_t>(labels.size()) + 1;
	for (uint32_t r = 0; r < repeats; ++r)
	{
		vk::CommandBufferSubmitInfo info{.commandBuffer = *cb};
		ctx_.queue.submit2(vk::SubmitInfo2{.commandBufferInfoCount = 1, .pCommandBufferInfos = &info}, nullptr);
		ctx_.queue.waitIdle();
		auto [res, ts] = pool.getResults<uint64_t>(0, n, n * sizeof(uint64_t), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		if (res != vk::Result::eSuccess)
			return {};
		for (size_t k = 0; k < labels.size(); ++k)
			std::ranges::find(phases, labels[k], &PhaseTime::label)->us += double(ts[k + 1] - ts[k]) * period * 1e-3 / repeats;
	}
	return phases;
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
	dispatchGrid(cb, Kernel::DiagPartial);
	vkc::computeBarrier(cb);
	bind(cb, Kernel::DiagFinal);
	cb.dispatch(1, 1, 1);
	bind(cb, Kernel::Probes);
	cb.dispatch(1, 1, 1);
	bind(cb, Kernel::Profile);
	cb.dispatch(MAX_PROFILE / 64, 1, 1);

	vkc::memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
	cb.copyBuffer(scalars_.handle(), readback_[slot].handle(),
	              vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = S_COUNT * sizeof(uint32_t)});
	cb.copyBuffer(profileOut_.handle(), profileReadback_[slot].handle(),
	              vk::BufferCopy{.srcOffset = 0, .dstOffset = 0, .size = MAX_PROFILE * sizeof(ProfilePoint)});
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
	                .t        = std::ldexp(double(s[S_TIME_HI]), -8) + std::ldexp(double(s[S_TIME_LO]), -40),
	                .n_steps  = s[S_NSTEPS],
	                .volume   = f(S_VOLUME),
	                .wetCells = f(S_WET_CELLS),
	                .hMax     = f(S_HMAX),
	                .uMax     = f(S_UMAX),
	                .limitedCells = s[S_LIMITED],
	                .clampCells   = s[S_CLAMP_N],
	                .clampMax     = f(S_CLAMP_MAX),
	                .gwVolume     = f(S_GW_VOLUME),
	                .gwInfil      = f(S_GW_INFIL),
	                .gwExfil      = f(S_GW_EXFIL),
	                .gwRate       = f(S_GW_RATE),
	                .gwDiffusivity = f(S_GW_D),
	                .srcNet       = f(S_SRC_NET),
	                .srcRain      = f(S_SRC_RAIN)};
	for (uint32_t p = 0; p < MAX_PROBES; ++p)
		r.probes[p] = {f(S_PROBES + 4 * p), f(S_PROBES + 4 * p + 1), f(S_PROBES + 4 * p + 2), f(S_PROBES + 4 * p + 3),
		               f(S_GW_PROBES + 2 * p), f(S_GW_PROBES + 2 * p + 1)};
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
		p[1 + k] = std::min<uint32_t>(cells[k], static_cast<uint32_t>(this->cells()) - 1);
}

void Solver::setProfile(std::vector<uint32_t> const &cells)
{
	// read by in-flight readbacks: only written with the device idle
	ctx_.device.waitIdle();
	auto    *p = static_cast<uint32_t *>(profileCells_.mapped());
	uint32_t n = static_cast<uint32_t>(std::min<size_t>(cells.size(), MAX_PROFILE));
	p[0]       = n;
	for (uint32_t k = 0; k < n; ++k)
		p[1 + k] = std::min<uint32_t>(cells[k], static_cast<uint32_t>(this->cells()) - 1);
}

std::vector<ProfilePoint> Solver::readProfile(uint32_t slot) const
{
	uint32_t n = std::min(static_cast<uint32_t const *>(profileCells_.mapped())[0], MAX_PROFILE);
	auto    *p = static_cast<ProfilePoint const *>(profileReadback_[slot].mapped());
	return {p, p + n};
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
	size_t const       n  = cells();
	auto               q4 = Q_.download<float>(ctx_, n * 4);
	std::vector<float> q3(n * 3);
	for (size_t k = 0; k < n; ++k)
		for (int c = 0; c < 3; ++c)
			q3[3 * k + c] = q4[4 * k + c];
	return q3;
}

std::vector<float> Solver::bedSync() const
{
	return zb_.download<float>(ctx_, cells());
}

std::vector<float> Solver::groundwaterSync() const
{
	return gwOn_ ? gw_.download<float>(ctx_, cells() * 4) : std::vector<float>{};
}

std::vector<float> Solver::sourcesSync() const
{
	if (!srcOn_)
		return {};
	// signed 64-bit fixed point (hi: 2^-8 m, lo: 2^-40 m) per quantity, see groundwater.slang
	auto               raw = srcCum_.download<uint32_t>(ctx_, cells() * 4);
	std::vector<float> out(cells() * 2);
	for (size_t k = 0; k < out.size(); ++k)
	{
		auto v = int64_t((uint64_t(raw[2 * k]) << 32) | raw[2 * k + 1]);
		out[k] = static_cast<float>(std::ldexp(double(v), -40));
	}
	return out;
}

void Solver::setChronologies(CaseData const &data)
{
	ctx_.device.waitIdle();
	bool on = data.infiltration.active() || data.rain.active();
	if (on && !srcOn_)
	{
		// first chronology of the case: per-cell buffers at full size (cumulated depths start at 0)
		vk::DeviceSize const n = cells();
		srcZones_ = vkc::Buffer::storage(ctx_, n * sizeof(uint32_t), "src_zones");
		srcCum_   = vkc::Buffer::storage(ctx_, n * 4 * sizeof(uint32_t), "src_cum", vk::BufferUsageFlagBits::eVertexBuffer);
		if (!gwOn_)
			diagGw_ = vkc::Buffer::storage(ctx_, vk::DeviceSize(2) * tileCount() * 4 * sizeof(float), "diag_vertical");
		ctx_.oneShot([&](vk::raii::CommandBuffer &cb) {
			for (auto *b : {&srcZones_, &srcCum_, &diagGw_})
				cb.fillBuffer(b->handle(), 0, vk::WholeSize, 0);
		});
	}
	srcOn_ = on;
	if (on)
		uploadChronologies(data);        // rewrites the descriptors
	else
		writeDescriptors();
	// new descriptors invalidate every command buffer that bound the set; the step also changes (vertical kernel)
	recordStep();
	for (uint32_t s = 0; s < kReadbackSlots; ++s)
		recordReadback(s);
}

// Zone maps (infiltration | rain << 16) and the two tables, rates divided by the zone areas (m3/s -> m/s, as
// wolfgpu's _timings_as_gpu_array). Zones beyond the chronology's columns, or without cells, get nothing.
void Solver::uploadChronologies(CaseData const &data)
{
	ctx_.device.waitIdle();
	size_t const                                n = cells();
	std::array<CaseData::Chronology const *, 2> tables{&data.infiltration, &data.rain};
	std::vector<uint32_t>                       zones(n, 0);
	std::vector<float>                          header(8, 0.0f), body;
	size_t                                      rates = 0;
	for (int k = 0; k < 2; ++k)
	{
		auto const &ch = *tables[k];
		if (!ch.active())
			continue;
		size_t const          nz = ch.zoneCount();
		std::vector<uint32_t> count(nz, 0);
		for (size_t c = 0; c < n; ++c)
			if (uint32_t z = ch.zones[c]; z >= 1 && z <= nz)
			{
				zones[c] |= z << (16 * k);
				++count[z - 1];
			}
		header[4 * k + 0] = float(ch.t.size());
		header[4 * k + 1] = float(nz);
		header[4 * k + 2] = ch.lerp ? 1.0f : 0.0f;
		header[4 * k + 3] = float(8 + body.size());
		for (size_t l = 0; l < ch.t.size(); ++l)
		{
			body.push_back(static_cast<float>(ch.t[l]));
			for (size_t z = 0; z < nz; ++z)
				body.push_back(count[z] > 0 ? static_cast<float>(ch.q[l][z] / (count[z] * data.dx() * data.dy())) : 0.0f);
		}
		rates += nz;
	}
	std::vector<float> all = header;
	all.insert(all.end(), body.begin(), body.end());
	chrono_   = vkc::Buffer::storage(ctx_, all.size() * sizeof(float), "chrono");
	srcRates_ = vkc::Buffer::storage(ctx_, std::max<size_t>(rates, 1) * sizeof(float), "src_rates");
	ctx_.oneShot([&](vk::raii::CommandBuffer &cb) { cb.fillBuffer(srcRates_.handle(), 0, vk::WholeSize, 0); });
	chrono_.upload(ctx_, all.data(), all.size() * sizeof(float));
	srcZones_.upload(ctx_, zones.data(), n * sizeof(uint32_t));
	writeDescriptors();        // new table buffers
}

}        // namespace swe
