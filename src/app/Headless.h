#pragma once
#include "util/Cli.h"

#include <string>

namespace swe {
namespace vkc { class Context; }

// Batch run: load a case, advance --steps steps, optionally dump Q/zb/meta to --dump. Returns an exit code.
int runHeadless(Options const &opts, vkc::Context const &ctx);

// Geographic import without GPU: --import xmin,ymin,xmax,ymax --cell m [--library dir]... [--name n] [--dump dir]
// Builds the bed from the configured sources, saves the case (assets) and a GeoTIFF + report in --dump.
int runImport(Options const &opts);

// --assets, else ./assets, else the source tree
std::string resolveAssetDir(std::string const &requested);

}        // namespace swe
