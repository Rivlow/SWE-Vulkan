#include "sim/Import.h"

#include "cases/Cases.h"
#include "util/Png.h"

#include <filesystem>

namespace swe {

CaseData saveImport(geo::ImportResult const &result, geo::ImportRequest const &request, geo::Image const &orthophoto,
                    std::string const &assetDir)
{
	auto           data = geo::TerrainBuilder::toCase(result, request);
	ImportMetadata meta{.x0 = result.bed.x0, .y0 = result.bed.y0, .cell = result.bed.cell, .epsg = result.bed.epsg,
	                    .attributions = result.attributions};
	for (auto const &r : result.sources)
		if (r.used)
			meta.sources.push_back(r.name);
	if (!orthophoto.empty())
	{
		meta.orthoFile = "data/" + data.name + "_ortho.png";
		writePng(std::filesystem::path(assetDir) / meta.orthoFile, orthophoto.rgba.data(), orthophoto.width, orthophoto.height, false);
	}
	saveImportedCase(data, meta, assetDir);
	return data;
}

}        // namespace swe
