#pragma once
#include "util/Cli.h"

#include <string>

namespace swe {
namespace vkc { class Context; }

// Batch run: load a case, advance --steps steps, optionally dump Q/zb/meta to --dump. Returns an exit code.
int runHeadless(Options const &opts, vkc::Context const &ctx);

// --assets, else ./assets, else the source tree
std::string resolveAssetDir(std::string const &requested);

}        // namespace swe
