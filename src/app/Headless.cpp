#include "app/Headless.h"

#include "cases/Cases.h"
#include "cases/Npy.h"
#include "geo/TerrainBuilder.h"
#include "solver/Solver.h"
#include "util/Log.h"
#include "vk/Context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace swe {

std::string resolveAssetDir(std::string const &requested)
{
	if (!requested.empty())
		return requested;
	if (std::filesystem::exists("assets"))
		return "assets";
	return SWE_ASSET_DIR;
}

int runHeadless(Options const &opts, vkc::Context const &ctx)
{
	auto data = makeCase(opts.caseName, resolveAssetDir(opts.assetDir));
	if (!opts.scheme.empty())
		data.scheme = opts.scheme;

	Solver solver(ctx);
	solver.load(data);

	auto t0 = std::chrono::steady_clock::now();
	solver.run(static_cast<uint32_t>(std::max(opts.steps, 0)));
	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

	auto s = solver.scalarsSync();
	log::info("{} steps in {:.3f} s ({:.1f} us/step, {:.1f} Mcell-steps/s): t = {:.6f} s, dt = {:.4e} s, s_max = {:.4f} m/s",
	          s.n_steps, seconds, opts.steps > 0 ? 1e6 * seconds / opts.steps : 0.0,
	          seconds > 0 ? double(data.cells()) * opts.steps / seconds * 1e-6 : 0.0, s.t, s.dt, s.s_max);

	if (!opts.dumpDir.empty())
	{
		std::filesystem::path dir(opts.dumpDir);
		std::filesystem::create_directories(dir);
		npy::save(dir / "Q.npy", {data.nx, data.ny, 3}, solver.stateSync());
		npy::save(dir / "zb.npy", {data.nx, data.ny}, solver.bedSync());
		nlohmann::json meta{{"source", "swe-vulkan"},
		                    {"case", data.name},
		                    {"scheme", solver.schemeName()},
		                    {"steps", s.n_steps},
		                    {"nx", data.nx},
		                    {"ny", data.ny},
		                    {"t", s.t},
		                    {"dt", s.dt},
		                    {"s_max", s.s_max},
		                    {"device", ctx.properties.deviceName.data()}};
		std::ofstream(dir / "meta.json") << meta.dump(2) << '\n';
		log::info("dumped Q.npy, zb.npy, meta.json to {}", dir.string());
	}

	bool finite = std::isfinite(s.t) && std::isfinite(s.dt);
	if (!finite)
		log::error("non-finite time step: the simulation blew up");
	return finite ? 0 : 1;
}

int runImport(Options const &opts)
{
	geo::BBox box;
	if (sscanf_s(opts.importBox.c_str(), "%lf,%lf,%lf,%lf", &box.xmin, &box.ymin, &box.xmax, &box.ymax) != 4 || box.empty())
	{
		log::error("--import expects xmin,ymin,xmax,ymax in Lambert 72 metres");
		return 1;
	}
	geo::init();
	geo::LocalLibrary library;
	for (auto const &d : opts.libraryDirs)
		library.addFolder(d);
	if (!opts.libraryDirs.empty())
		library.rescan();
	geo::TerrainBuilder builder(library);
	geo::Progress       progress;

	log::info("probe {}x{} m at {} m", box.width(), box.height(), opts.importCell);
	for (auto const &r : builder.probe(box, progress))
		log::info("  {:<40} footprint {:5.1f} %  data {}{}", r.name, 100.0 * r.footprint,
		          r.data < 0 ? std::string("-") : std::format("{:5.1f} %", 100.0 * r.data), r.error.empty() ? "" : "  (" + r.error + ")");

	geo::ImportRequest req{.box = box, .cell = opts.importCell, .name = opts.importName, .label = opts.importName};
	std::string        error;
	auto               t0  = std::chrono::steady_clock::now();
	auto               res = builder.build(req, progress, &error);
	if (res.bed.v.empty())
	{
		log::error("import failed: {}", error);
		return 1;
	}
	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	log::info("bed {}x{} cells in {:.1f} s, bathymetry on {:.2f} % of the cells, {} hole cell(s)", res.bed.nx, res.bed.ny, seconds,
	          100.0 * res.bathyCoverage, std::count(res.outside.begin(), res.outside.end(), uint8_t(1)));
	for (auto const &r : res.sources)
		log::info("  {:<40} {}{}", r.name, r.used ? "used" : "unused", r.error.empty() ? "" : "  (" + r.error + ")");
	for (auto const &a : res.attributions)
		log::info("  attribution: {}", a);

	auto           data = geo::TerrainBuilder::toCase(res, req);
	ImportMetadata meta{.x0 = res.bed.x0, .y0 = res.bed.y0, .cell = res.bed.cell, .attributions = res.attributions};
	for (auto const &r : res.sources)
		if (r.used)
			meta.sources.push_back(r.name);
	saveImportedCase(data, meta, resolveAssetDir(opts.assetDir));
	log::info("case '{}' saved in {}", data.name, resolveAssetDir(opts.assetDir));

	if (!opts.dumpDir.empty())
	{
		std::filesystem::path dir(opts.dumpDir);
		std::filesystem::create_directories(dir);
		std::string err;
		if (!geo::writeGeoTiff((dir / (opts.importName + ".tif")).string(), res.bed, &err))
			log::warn("GeoTIFF: {}", err);
		npy::save(dir / "zb_abs.npy", {res.bed.nx, res.bed.ny}, res.bed.v);
		log::info("dumped {}.tif and zb_abs.npy to {}", opts.importName, dir.string());
	}
	return 0;
}

}        // namespace swe
