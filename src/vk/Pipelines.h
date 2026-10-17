#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace swe::vkc {

class Context;

// Directory holding the compiled .spv files (bin/shaders next to the executable, else the build tree).
std::filesystem::path shaderDir();

vk::raii::ShaderModule loadShader(Context const &ctx, std::string const &name);        // name without ".spv"

vk::raii::Pipeline makeComputePipeline(Context const &ctx, vk::raii::ShaderModule const &module, char const *entry,
                                       vk::raii::PipelineLayout const &layout);

enum class Blend
{
	None,
	Alpha,              // src * a + dst * (1 - a)
	DualSourceMul,      // src0 + dst * src1 (water transparency)
};

// Graphics pipeline for dynamic rendering, no vertex input (vertices are fetched from buffers by
// SV_VertexID), dynamic viewport/scissor, no culling (meshes are seen from both sides).
struct GraphicsPipelineDesc
{
	vk::ShaderModule        vertexModule;
	char const             *vertexEntry   = "vertMain";
	vk::ShaderModule        fragmentModule;
	char const             *fragmentEntry = "fragMain";
	vk::PipelineLayout      layout;
	vk::Format              colorFormat   = vk::Format::eUndefined;
	vk::Format              depthFormat   = vk::Format::eUndefined;
	vk::SampleCountFlagBits samples       = vk::SampleCountFlagBits::e1;
	vk::PrimitiveTopology   topology      = vk::PrimitiveTopology::eTriangleList;
	vk::PolygonMode         polygonMode   = vk::PolygonMode::eFill;
	Blend                   blend         = Blend::None;
	bool                    depthTest     = true;
	bool                    depthWrite    = true;
	float                   depthBias     = 0.0f;        // constant bias (wireframe overlays)
	std::string             name;
};

vk::raii::Pipeline makeGraphicsPipeline(Context const &ctx, GraphicsPipelineDesc const &desc);

// Global memory barrier (synchronization2).
void memoryBarrier(vk::raii::CommandBuffer const &cb, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess);

// compute write -> compute read/write
inline void computeBarrier(vk::raii::CommandBuffer const &cb)
{
	memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
}

// transfer write -> compute read/write, or compute write -> transfer read/write
inline void transferToCompute(vk::raii::CommandBuffer const &cb)
{
	memoryBarrier(cb, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);
}
inline void computeToTransfer(vk::raii::CommandBuffer const &cb)
{
	memoryBarrier(cb, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	              vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
}

inline uint32_t divUp(uint32_t n, uint32_t d) { return (n + d - 1) / d; }

}        // namespace swe::vkc
