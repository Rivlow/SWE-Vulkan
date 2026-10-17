#pragma once
#include <vulkan/vulkan_raii.hpp>

#include <filesystem>
#include <string>

namespace swe::vkc {

class Context;

// Directory holding the compiled .spv files (bin/shaders next to the executable, else the build tree).
std::filesystem::path shaderDir();

vk::raii::ShaderModule loadShader(Context const &ctx, std::string const &name);        // name without ".spv"

vk::raii::Pipeline makeComputePipeline(Context const &ctx, vk::raii::ShaderModule const &module, char const *entry,
                                       vk::raii::PipelineLayout const &layout);

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
