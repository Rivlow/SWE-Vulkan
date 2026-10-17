#include "vk/Pipelines.h"

#include "vk/Context.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#	include <windows.h>
#endif

namespace swe::vkc {

std::filesystem::path shaderDir()
{
#ifdef _WIN32
	wchar_t exe[MAX_PATH];
	if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0)
	{
		auto dir = std::filesystem::path(exe).parent_path() / "shaders";
		if (std::filesystem::exists(dir))
			return dir;
	}
#endif
	return SWE_SHADER_DIR;
}

vk::raii::ShaderModule loadShader(Context const &ctx, std::string const &name)
{
	auto          path = shaderDir() / (name + ".spv");
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file)
		throw std::runtime_error("cannot open shader " + path.string());
	auto                  size = static_cast<size_t>(file.tellg());
	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(size));
	vk::raii::ShaderModule module(ctx.device, vk::ShaderModuleCreateInfo{.codeSize = size, .pCode = code.data()});
	ctx.setName(module, name);
	return module;
}

vk::raii::Pipeline makeComputePipeline(Context const &ctx, vk::raii::ShaderModule const &module, char const *entry,
                                       vk::raii::PipelineLayout const &layout)
{
	vk::ComputePipelineCreateInfo info{
	    .stage  = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *module, .pName = entry},
	    .layout = *layout};
	vk::raii::Pipeline pipeline(ctx.device, nullptr, info);
	ctx.setName(pipeline, entry);
	return pipeline;
}

void memoryBarrier(vk::raii::CommandBuffer const &cb, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
	vk::MemoryBarrier2 barrier{.srcStageMask = srcStage, .srcAccessMask = srcAccess, .dstStageMask = dstStage, .dstAccessMask = dstAccess};
	cb.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

}        // namespace swe::vkc
