#pragma once

namespace swe {
namespace vkc { class Context; }

// M0 sanity check: one compute dispatch, readback, host-side verification.
bool runSelfTest(vkc::Context const &ctx);

}        // namespace swe
