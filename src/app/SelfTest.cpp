#include "app/SelfTest.h"

#include "util/Log.h"
#include "vk/Buffer.h"
#include "vk/Context.h"
#include "vk/Pipelines.h"

namespace swe {

// Dispatches shaders/selftest.slang:fill (buf[i] = 2i + 1) and checks the sum on the host: sum = n^2.
bool runSelfTest(vkc::Context const &ctx)
{
	constexpr uint32_t n = 1'000'003;        // not a multiple of the workgroup size on purpose

	auto buffer = vkc::Buffer::storage(ctx, n * sizeof(uint32_t), "selftest");

	vk::DescriptorSetLayoutBinding binding{.binding         = 0,
	                                       .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                       .descriptorCount = 1,
	                                       .stageFlags      = vk::ShaderStageFlagBits::eCompute};
	vk::raii::DescriptorSetLayout setLayout(ctx.device, {.bindingCount = 1, .pBindings = &binding});
	vk::PushConstantRange         pcRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(uint32_t)};
	vk::raii::PipelineLayout      layout(ctx.device, {.setLayoutCount         = 1,
	                                                  .pSetLayouts            = &*setLayout,
	                                                  .pushConstantRangeCount = 1,
	                                                  .pPushConstantRanges    = &pcRange});

	vk::DescriptorPoolSize   poolSize{vk::DescriptorType::eStorageBuffer, 1};
	vk::raii::DescriptorPool pool(ctx.device, {.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
	                                           .maxSets       = 1,
	                                           .poolSizeCount = 1,
	                                           .pPoolSizes    = &poolSize});
	auto                     sets = vk::raii::DescriptorSets(ctx.device, {.descriptorPool     = *pool,
	                                                                      .descriptorSetCount = 1,
	                                                                      .pSetLayouts        = &*setLayout});
	auto                     info = buffer.info();
	ctx.device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet          = *sets[0],
	                                                       .dstBinding      = 0,
	                                                       .descriptorCount = 1,
	                                                       .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                                       .pBufferInfo     = &info},
	                                nullptr);

	auto module   = vkc::loadShader(ctx, "selftest");
	auto pipeline = vkc::makeComputePipeline(ctx, module, "fill", layout);

	ctx.oneShot([&](vk::raii::CommandBuffer &cb) {
		cb.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
		cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, *sets[0], nullptr);
		cb.pushConstants<uint32_t>(*layout, vk::ShaderStageFlagBits::eCompute, 0, n);
		cb.dispatch(vkc::divUp(n, 256), 1, 1);
	});

	auto     data = buffer.download<uint32_t>(ctx, n);
	uint64_t sum  = 0;
	for (auto v : data)
		sum += v;
	uint64_t expected = uint64_t(n) * n;
	bool     ok       = sum == expected;
	log::info("selftest: sum = {} (expected {}) -> {}", sum, expected, ok ? "OK" : "FAILED");
	return ok;
}

}        // namespace swe
