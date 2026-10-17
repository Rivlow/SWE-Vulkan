#pragma once
#include "cases/Case.h"
#include "solver/RungeKutta.h"
#include "solver/SolverParams.h"
#include "vk/Buffer.h"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace swe {

namespace vkc { class Context; }

// GPU shallow-water solver: owns the state buffers and the compute pipelines of shaders/solver/solver.slang.
// One time step is a pre-recorded command buffer (simultaneous use) that can be submitted many times
// in a row; scalars (t, dt, s_max...) are copied to a host-visible slot for a lag-free readback.
class Solver
{
  public:
	static constexpr uint32_t kReadbackSlots = 2;        // = frames in flight

	explicit Solver(vkc::Context const &ctx);

	// (Re)allocates everything for a case and uploads its initial state. Waits for the device.
	void load(CaseData const &data);
	// Restores the initial state of the loaded case (t = 0). Waits for the device.
	void reset();
	// Switches the Runge-Kutta scheme (re-records the step). Waits for the device.
	void setScheme(std::string const &name);
	// Runtime tuning (cfl, manning n, boundary conditions); applied on the next submission.
	void setCfl(double cfl);
	void setManning(double n);
	void setBoundary(int side, BoundaryCondition const &bc);
	// Cells (index i * ny + j) whose state is read back each frame, at most MAX_PROBES. Waits for the device.
	void setProbes(std::vector<uint32_t> const &cells);

	// Interactive editing (waits for the device):
	// water block in the inclusive cell rectangle, applied to the live state and, if `alsoInitial`, to h0
	void editWater(uint32_t i0, uint32_t j0, uint32_t i1, uint32_t j1, WaterEdit mode, float value, bool alsoInitial);
	// replaces the per-cell zones (CaseData::zones / zoneMask) and uploads them
	void setZones(std::vector<CaseData::ZoneDef> const &zones, std::vector<uint8_t> const &mask);

	// Appends `steps` time steps followed by a scalar readback into `slot` to a submission batch.
	void appendSteps(std::vector<vk::CommandBufferSubmitInfo> &batch, uint32_t steps, uint32_t slot);
	// Scalars copied into `slot` (valid once the submission that filled it has completed).
	SolverScalars readback(uint32_t slot) const;

	// Headless helpers (synchronous)
	void          run(uint32_t steps);
	SolverScalars scalarsSync() const;
	std::vector<float> stateSync() const;        // (nx, ny, 3) = h, hu, hv
	std::vector<float> bedSync() const;          // (nx, ny)

	CaseData const       &caseData() const { return case_; }
	std::string const    &schemeName() const { return schemeName_; }
	SolverParams const   &params() const { return params_; }
	bool                  loaded() const { return case_.nx != 0; }

	// For the renderer (read-only use of the solver set as an extra descriptor set)
	vk::DescriptorSetLayout setLayout() const { return *setLayout_; }
	vk::DescriptorSet       set() const { return *set_; }
	vkc::Buffer const      &stateBuffer() const { return Q_; }
	vkc::Buffer const      &bedBuffer() const { return zb_; }

  private:
	void createLayout();
	void createPipelines();
	void allocate();
	void writeDescriptors();
	void uploadInitialState();
	void updateParams();
	void recordStep();
	void recordReadback(uint32_t slot);
	void dispatchGrid(vk::raii::CommandBuffer const &cb, char const *entry) const;
	void bind(vk::raii::CommandBuffer const &cb, char const *entry) const;
	void pushStage(vk::raii::CommandBuffer &cb, StagePC const &pc) const;

	vkc::Context const &ctx_;

	CaseData       case_;
	std::string    schemeName_ = "ssprk2";
	ButcherTableau tableau_;
	SolverParams   params_{};

	vk::raii::DescriptorSetLayout                setLayout_      = nullptr;
	vk::raii::PipelineLayout                     pipelineLayout_ = nullptr;
	vk::raii::DescriptorPool                     pool_           = nullptr;
	vk::raii::DescriptorSet                      set_            = nullptr;
	vk::raii::ShaderModule                       module_         = nullptr;
	std::map<std::string, vk::raii::Pipeline>    pipelines_;

	// paramsBuf_ is host-visible: updateParams() waits for the device, then writes it directly
	vkc::Buffer paramsBuf_, scalars_, zb_, Q_, Q_old_, K_, F_L_, F_R_, G_D_, G_U_, S_b_, sink_, inflow_;
	vkc::Buffer probeBuf_, diag_;        // probe cell list (host-visible), per-workgroup diagnostics
	vkc::Buffer zones_;                  // MAX_ZONES Zone entries (sink_ holds the per-cell zone index)
	void        uploadZones();
	std::array<vkc::Buffer, kReadbackSlots> readback_;

	vk::raii::CommandBuffer                             stepCmd_ = nullptr;
	std::array<vk::raii::CommandBuffer, kReadbackSlots> readbackCmd_{nullptr, nullptr};
};

}        // namespace swe
