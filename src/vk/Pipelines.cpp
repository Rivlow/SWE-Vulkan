#include "vk/Pipelines.h"

#include "vk/Context.h"

#include <array>
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

vk::raii::Pipeline makeGraphicsPipeline(Context const &ctx, GraphicsPipelineDesc const &d)
{
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = d.vertexModule, .pName = d.vertexEntry},
	    vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = d.fragmentModule, .pName = d.fragmentEntry}};

	vk::PipelineVertexInputStateCreateInfo   vertexInput{};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = d.topology};
	vk::PipelineViewportStateCreateInfo      viewport{.viewportCount = 1, .scissorCount = 1};
	vk::PipelineRasterizationStateCreateInfo raster{.polygonMode             = d.polygonMode,
	                                                .cullMode                = vk::CullModeFlagBits::eNone,
	                                                .frontFace               = vk::FrontFace::eCounterClockwise,
	                                                .depthBiasEnable         = d.depthBias != 0.0f,
	                                                .depthBiasConstantFactor = d.depthBias,
	                                                .lineWidth               = 1.0f};
	vk::PipelineMultisampleStateCreateInfo   multisample{.rasterizationSamples = d.samples};
	vk::PipelineDepthStencilStateCreateInfo  depth{.depthTestEnable  = d.depthTest,
	                                               .depthWriteEnable = d.depthWrite,
	                                               .depthCompareOp   = vk::CompareOp::eLessOrEqual};

	vk::PipelineColorBlendAttachmentState attachment{
	    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
	                      vk::ColorComponentFlagBits::eA};
	if (d.blend == Blend::Alpha)
	{
		attachment.blendEnable         = true;
		attachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
		attachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.colorBlendOp        = vk::BlendOp::eAdd;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
		attachment.alphaBlendOp        = vk::BlendOp::eAdd;
	}
	else if (d.blend == Blend::DualSourceMul)
	{
		attachment.blendEnable         = true;
		attachment.srcColorBlendFactor = vk::BlendFactor::eOne;
		attachment.dstColorBlendFactor = vk::BlendFactor::eSrc1Color;
		attachment.colorBlendOp        = vk::BlendOp::eAdd;
		attachment.srcAlphaBlendFactor = vk::BlendFactor::eZero;
		attachment.dstAlphaBlendFactor = vk::BlendFactor::eOne;
		attachment.alphaBlendOp        = vk::BlendOp::eAdd;
	}
	vk::PipelineColorBlendStateCreateInfo blend{.attachmentCount = 1, .pAttachments = &attachment};

	std::array<vk::DynamicState, 2>    dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
	                                           .pDynamicStates    = dynamicStates.data()};

	vk::PipelineRenderingCreateInfo rendering{.colorAttachmentCount    = 1,
	                                          .pColorAttachmentFormats = &d.colorFormat,
	                                          .depthAttachmentFormat   = d.depthFormat};

	vk::GraphicsPipelineCreateInfo info{.pNext               = &rendering,
	                                    .stageCount          = static_cast<uint32_t>(stages.size()),
	                                    .pStages             = stages.data(),
	                                    .pVertexInputState   = &vertexInput,
	                                    .pInputAssemblyState = &inputAssembly,
	                                    .pViewportState      = &viewport,
	                                    .pRasterizationState = &raster,
	                                    .pMultisampleState   = &multisample,
	                                    .pDepthStencilState  = &depth,
	                                    .pColorBlendState    = &blend,
	                                    .pDynamicState       = &dynamic,
	                                    .layout              = d.layout};
	vk::raii::Pipeline pipeline(ctx.device, nullptr, info);
	if (!d.name.empty())
		ctx.setName(pipeline, d.name);
	return pipeline;
}

void memoryBarrier(vk::raii::CommandBuffer const &cb, vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                   vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
{
	vk::MemoryBarrier2 barrier{.srcStageMask = srcStage, .srcAccessMask = srcAccess, .dstStageMask = dstStage, .dstAccessMask = dstAccess};
	cb.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

}        // namespace swe::vkc
