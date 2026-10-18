#pragma once
#include "cases/Case.h"
#include "solver/RungeKutta.h"
#include "solver/SolverParams.h"
#include "vk/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace swe {

namespace vkc { class Context; }

// GPU shallow-water solver: owns the state buffers and the compute pipelines of shaders/solver/solver.slang.
// One time step is a pre-recorded command buffer (simultaneous use) that can be submitted many times
// in a row; scalars (t, dt, s_max...) are copied to a host-visible slot for a lag-free readback.
// The solver does not own the case: every call takes the CaseData it applies (see sim/Session).
class Solver
{
  public:
	static constexpr uint32_t kReadbackSlots = 2;        // = frames in flight

	explicit Solver(vkc::Context const &ctx);

	// (Re)allocates everything for the grid of `data`, uploads its bed, zones and initial state and applies its
	// parameters. Waits for the device.
	void load(CaseData const &data);
	// Restores the initial state of `data` (same grid as loaded), t = 0. Waits for the device.
	void reset(CaseData const &data);
	// Applies the runtime parameters of `data` (cfl, Manning, boundary conditions, scheme, drain passes);
	// re-records the step when its structure changes. Waits for the device.
	void configure(CaseData const &data);
	// Uploads the per-cell zones of `data` (zones / zoneMask). Waits for the device.
	void uploadZones(CaseData const &data);
	// Uploads the infiltration / rain chronologies of `data` (zone maps and tables), turning them on or off. The
	// cumulated depths are kept. Waits for the device.
	void setChronologies(CaseData const &data);
	// Water block in a cell rectangle of the live state. Waits for the device.
	void editWater(CellRect const &rect, WaterEdit mode, float value);
	// Cells (index i * ny + j) whose state is read back each frame, at most MAX_PROBES. Waits for the device.
	void setProbes(std::vector<uint32_t> const &cells);
	// Cells of the profile read back each frame (at most MAX_PROFILE). Waits for the device.
	void setProfile(std::vector<uint32_t> const &cells);
	// Profile copied into `slot` (valid once the submission that filled it has completed)
	std::vector<ProfilePoint> readProfile(uint32_t slot) const;

	// Appends `steps` time steps followed by a scalar readback into `slot` to a submission batch.
	void appendSteps(std::vector<vk::CommandBufferSubmitInfo> &batch, uint32_t steps, uint32_t slot);
	// Scalars copied into `slot` (valid once the submission that filled it has completed).
	SolverScalars readback(uint32_t slot) const;

	// Headless helpers (synchronous)
	void               run(uint32_t steps);
	SolverScalars      scalarsSync() const;
	std::vector<float> stateSync() const;        // (nx, ny, 3) = h, hu, hv
	std::vector<float> bedSync() const;          // (nx, ny)
	std::vector<float> groundwaterSync() const;  // (nx, ny, 4) = h_gw, exchange rate, cumulated infiltration / exfiltration (empty without)
	std::vector<float> sourcesSync() const;      // (nx, ny, 2) = net depth added by the chronologies, rain depth (empty without)
	bool               verticalOn() const { return gwOn_ || srcOn_; }

	std::string const  &schemeName() const { return schemeName_; }
	SolverParams const &params() const { return params_; }

	// For the renderer (read-only use of the solver set as an extra descriptor set)
	vk::DescriptorSetLayout setLayout() const { return *setLayout_; }
	vk::DescriptorSet       set() const { return *set_; }

	// GPU time of each phase of a step (µs, averaged over `repeats` steps that advance the simulation; phases
	// repeated within a step, such as the stages, are summed). Empty without timestamp support.
	struct PhaseTime
	{
		std::string label;
		double      us = 0.0;
	};
	std::vector<PhaseTime> profileStep(uint32_t repeats);

	// Computes every tile, wet or dry (reference for the active-tile scheme, which must give the same bits)
	void setAllTilesActive(bool on) { allTiles_ = on; }
	// Tiles computed by the last step (read back synchronously, for tests and benchmarks)
	uint32_t activeTilesSync() const;
	uint32_t tileCount() const { return tilesX_ * tilesY_; }

	enum class Kernel : uint32_t
	{
		BuildTiles,
		ReduceSmax,
		FinalizeDt,
		FluxX,
		FluxY,
		BcFluxes,
		DrainAlpha,
		EulerStage,
		DiagPartial,
		DiagFinal,
		Probes,
		SetWater,
		GwAlpha,
		GwLateral,
		Vertical,        // chronologies + groundwater exchange
		Profile,
		Count,
	};

  private:
	void createLayout();
	void createPipelines();
	void allocate();
	void writeDescriptors();
	void reduceSmax();
	using Marker = std::function<void(char const *)>;
	void recordStep();
	void recordStepCommands(vk::raii::CommandBuffer const &cb, Marker const &mark) const;
	void recordReadback(uint32_t slot);
	void bind(vk::raii::CommandBuffer const &cb, Kernel k) const;
	void dispatchGrid(vk::raii::CommandBuffer const &cb, Kernel k) const;
	void dispatchActive(vk::raii::CommandBuffer const &cb, Kernel k) const;        // one workgroup per active tile
	void uploadTileSeeds(CaseData const &data);
	void uploadChronologies(CaseData const &data);        // zone maps + tables (sizes chrono_, srcRates_)
	void pushStage(vk::raii::CommandBuffer const &cb, StagePC const &pc) const;
	size_t cells() const { return size_t(nx_) * ny_; }

	vkc::Context const &ctx_;

	// grid and step structure of the loaded case
	uint32_t     nx_ = 0, ny_ = 0;
	uint32_t     tilesX_ = 0, tilesY_ = 0;        // 16 x 16 cell tiles
	std::array<BcKind, 4> seededBc_{};            // boundary kinds the tile seeds were computed for
	bool                  allTiles_ = false;
	std::string  schemeName_ = "ssprk2";
	TimeScheme   scheme_;
	int          drainPasses_ = 0;
	SolverParams params_{};

	vk::raii::DescriptorSetLayout setLayout_      = nullptr;
	vk::raii::PipelineLayout      pipelineLayout_ = nullptr;
	vk::raii::DescriptorPool      pool_           = nullptr;
	vk::raii::DescriptorSet       set_            = nullptr;
	vk::raii::ShaderModule        module_         = nullptr;
	std::vector<vk::raii::Pipeline> pipelines_;        // indexed by Kernel

	// paramsBuf_ and probeBuf_ are host-visible: written directly once the device is idle
	vkc::Buffer paramsBuf_, probeBuf_, profileCells_;
	vkc::Buffer profileOut_;                                   // device, filled by the readback
	std::array<vkc::Buffer, kReadbackSlots> profileReadback_;  // host copies per slot
	vkc::Buffer scalars_, zb_, Q_, Q_old_, F_L_, F_R_, G_D_, G_U_, inflow_, diag_;
	std::array<vkc::Buffer, 2> alpha_;        // drain limiter factors, Jacobi ping-pong
	vkc::Buffer massFlux_;                   // drain limiter: per-cell face mass fluxes (gathered by pass 0)
	// active tiles: wet stamps, list, indirect arguments, seeds (see solver.slang)
	vkc::Buffer tileWet_, tileList_, tileArgs_, tileSeed_;
	vkc::Buffer cond_, zones_;               // per-cell zone index + 1, MAX_ZONES Zone entries
	// groundwater: soil (kv, kh, Sy, z_sub), state, outflow factor + lateral change, diagnostics (1 element when off)
	vkc::Buffer soil_, gw_, gwAux_, diagGw_;
	// infiltration / rain chronologies: zone maps, tables, rates of the step, cumulated depths (1 element when off)
	vkc::Buffer srcZones_, chrono_, srcRates_, srcCum_;
	bool        srcOn_ = false;
	bool        gwOn_ = false, gwLateral_ = false;
	std::array<vkc::Buffer, kReadbackSlots> readback_;

	vk::raii::CommandBuffer                             stepCmd_ = nullptr;
	std::array<vk::raii::CommandBuffer, kReadbackSlots> readbackCmd_{nullptr, nullptr};
};

}        // namespace swe
