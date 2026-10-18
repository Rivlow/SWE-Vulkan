#pragma once
#include "cases/Case.h"
#include "geo/TerrainBuilder.h"
#include "geo/Tiles.h"

#include <string>

namespace swe {

// Saves a finished geographic import as a case of the asset directory: assets/cases/<name>.json,
// assets/data/<name>.npz and, when `orthophoto` is not empty, assets/data/<name>_ortho.png.
// Provenance (sources used, attributions) comes from `result`. Returns the case as saved.
CaseData saveImport(geo::ImportResult const &result, geo::ImportRequest const &request, geo::Image const &orthophoto,
                    std::string const &assetDir);

}        // namespace swe
