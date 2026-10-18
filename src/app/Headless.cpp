#include "app/Headless.h"

#include "cases/Cases.h"
#include "cases/Npy.h"
#include "geo/Carve.h"
#include "geo/TerrainBuilder.h"
#include "geo/Tiles.h"
#include "sim/Import.h"
#include "sim/Session.h"
#include "util/Log.h"
#include "vk/Context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

namespace swe {

void applyCaseEdits(Session &session, Options const &opts)
{
	for (auto const &w : opts.waterBlocks)
		session.editWater(w.rect, WaterEdit::Depth, w.depth, true);
	for (auto const &z : opts.zones)
		session.addZone(z.rect, z.kind, z.kind == ZoneKind::Level ? z.value - session.caseData().z_ref : z.value, z.name);
	for (auto const &z : opts.chronologyZones)
	{
		auto const &c    = session.caseData();
		CellRect    r    = z.rect;
		double      area = double(std::min(r.i1, c.nx - 1) - r.i0 + 1) * (std::min(r.j1, c.ny - 1) - r.j0 + 1) * c.dx() * c.dy();
		session.addChronologyZone(z.rain ? Session::ChronologyKind::Rain : Session::ChronologyKind::Infiltration, r,
		                          z.mmPerHour * area / 3.6e6);
	}
}

int runHeadless(Options const &opts, vkc::Context const &ctx)
{
	Session session(ctx);
	session.solver().setAllTilesActive(opts.allTiles);
	session.load(makeCase(opts.caseName, resolveAssetDir(opts.assetDir), opts.caseOverrides()));
	applyCaseEdits(session, opts);
	auto const &data   = session.caseData();
	auto       &solver = session.solver();

	auto t0 = std::chrono::steady_clock::now();
	solver.run(static_cast<uint32_t>(std::max(opts.steps, 0)));
	if (opts.untilTime > 0.0)
	{
		// batches of a quarter of the steps left at the current time step (dt may grow within a batch), single
		// steps at the end: stops within one step past the target
		for (auto s = solver.scalarsSync(); s.t < opts.untilTime && std::isfinite(s.t); s = solver.scalarsSync())
		{
			double left = s.dt > 0.0f ? (opts.untilTime - s.t) / s.dt : 1.0;        // no step yet: dt unknown
			solver.run(static_cast<uint32_t>(std::clamp(0.25 * left, 1.0, 2000.0)));
		}
	}
	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

	auto s = solver.scalarsSync();
	log::info("{} steps in {:.3f} s ({:.1f} us/step, {:.1f} Mcell-steps/s): t = {:.6f} s, dt = {:.4e} s, s_max = {:.4f} m/s",
	          s.n_steps, seconds, opts.steps > 0 ? 1e6 * seconds / opts.steps : 0.0,
	          seconds > 0 ? double(data.cells()) * opts.steps / seconds * 1e-6 : 0.0, s.t, s.dt, s.s_max);

	// water balance in double on the host (initial state from the case, final state read back)
	auto   state    = solver.stateSync();
	double cellArea = data.dx() * data.dy();
	double volume0 = 0.0, volume = 0.0, hMin = std::numeric_limits<double>::infinity();
	for (size_t k = 0; k < data.cells(); ++k)
	{
		volume0 += data.h0[k];
		volume += state[3 * k];
		hMin = std::min<double>(hMin, state[3 * k]);
	}
	volume0 *= cellArea;
	volume *= cellArea;
	log::info("drain limiter: {} passes, {} cell(s) limited on the last step, {} clamp(s) to h = 0 (max {:.3e} m), h min = {:.3e} m",
	          data.drain_passes, s.limitedCells, s.clampCells, s.clampMax, hMin);
	log::info("volume: {:.6f} m3 -> {:.6f} m3 ({:+.3e} relative)", volume0, volume,
	          volume0 > 0.0 ? (volume - volume0) / volume0 : 0.0);

	// groundwater: surface + aquifer is conserved (no rain, walls), infiltration / exfiltration explain the transfer
	auto   gwState = solver.groundwaterSync();
	double gwVolume0 = 0.0, gwVolume = 0.0, infil = 0.0, exfil = 0.0;
	if (!gwState.empty())
	{
		for (size_t k = 0; k < data.cells(); ++k)
		{
			double sy = data.gw.sy[k];
			gwVolume0 += sy * data.gw.h0[k];
			gwVolume += sy * gwState[4 * k];
			infil += gwState[4 * k + 2];
			exfil += gwState[4 * k + 3];
		}
		gwVolume0 *= cellArea;
		gwVolume *= cellArea;
		infil *= cellArea;
		exfil *= cellArea;
		log::info("groundwater: {:.6f} m3 -> {:.6f} m3, infiltrated {:.6f} m3, exfiltrated {:.6f} m3", gwVolume0, gwVolume, infil,
		          exfil);
	}
	// infiltration / rain chronologies: volume they added to the surface (net) and rain volume
	auto   srcState = solver.sourcesSync();
	double srcNet = 0.0, rainVolume = 0.0;
	for (size_t k = 0; k < srcState.size() / 2; ++k)
	{
		srcNet += srcState[2 * k];
		rainVolume += srcState[2 * k + 1];
	}
	srcNet *= cellArea;
	rainVolume *= cellArea;
	if (!srcState.empty())
		log::info("chronologies: rain {:.6f} m3, net volume added to the surface {:+.6f} m3", rainVolume, srcNet);
	if (!gwState.empty() || !srcState.empty())
	{
		// closed domain: surface + aquifer - what the chronologies added stays constant
		double total0 = volume0 + gwVolume0, total = volume + gwVolume - srcNet;
		log::info("balance surface + groundwater - chronologies: {:.6f} m3 -> {:.6f} m3 ({:+.3e} relative); "
		          "surface change - (exfil - infil) - added = {:+.3e} m3",
		          total0, total, (total - total0) / std::max(total0, volume + gwVolume), (volume - volume0) - (exfil - infil) - srcNet);
	}
	uint32_t activeTiles = solver.activeTilesSync();
	log::info("active tiles on the last step: {} / {} ({:.1f} %)", activeTiles, solver.tileCount(),
	          100.0 * activeTiles / std::max(solver.tileCount(), 1u));

	if (opts.profileSteps > 0)
	{
		auto   phases = solver.profileStep(static_cast<uint32_t>(opts.profileSteps));
		double total  = 0.0;
		for (auto const &p : phases)
			total += p.us;
		log::info("GPU time per step over {} more steps (phase = kernels + following barrier, stages summed):", opts.profileSteps);
		for (auto const &p : phases)
			log::info("  {:<50} {:8.2f} us  {:5.1f} %", p.label, p.us, total > 0 ? 100.0 * p.us / total : 0.0);
		log::info("  {:<50} {:8.2f} us", "total", total);
	}

	if (!opts.dumpDir.empty())
	{
		std::filesystem::path dir(opts.dumpDir);
		std::filesystem::create_directories(dir);
		npy::save(dir / "Q.npy", {data.nx, data.ny, 3}, state);
		npy::save(dir / "zb.npy", {data.nx, data.ny}, solver.bedSync());
		nlohmann::json meta{{"source", "swe-vulkan"},
		                    {"case", data.name},
		                    {"scheme", solver.schemeName()},
		                    {"drain_passes", data.drain_passes},
		                    {"steps", s.n_steps},
		                    {"nx", data.nx},
		                    {"ny", data.ny},
		                    {"t", s.t},
		                    {"dt", s.dt},
		                    {"s_max", s.s_max},
		                    {"volume0", volume0},
		                    {"volume", volume},
		                    {"h_min", hMin},
		                    {"limited_cells", s.limitedCells},
		                    {"clamp_cells", s.clampCells},
		                    {"clamp_max", s.clampMax},
		                    {"device", ctx.properties.deviceName.data()}};
		if (!gwState.empty())
		{
			npy::save(dir / "gw.npy", {data.nx, data.ny, 4}, gwState);
			meta["gw_volume0"] = gwVolume0;
			meta["gw_volume"]  = gwVolume;
			meta["gw_infiltrated"] = infil;
			meta["gw_exfiltrated"] = exfil;
		}
		if (!srcState.empty())
		{
			npy::save(dir / "sources.npy", {data.nx, data.ny, 2}, srcState);
			meta["sources_net"] = srcNet;
			meta["rain_volume"] = rainVolume;
		}
		std::ofstream(dir / "meta.json") << meta.dump(2) << '\n';
		log::info("dumped Q.npy, zb.npy{}{}, meta.json to {}", gwState.empty() ? "" : ", gw.npy", srcState.empty() ? "" : ", sources.npy",
		          dir.string());
	}

	bool finite = std::isfinite(s.t) && std::isfinite(s.dt);
	if (!finite)
		log::error("non-finite time step: the simulation blew up");
	return finite ? 0 : 1;
}

int runImport(Options const &opts)
{
	geo::init();
	geo::BBox box;
	int       epsg = geo::EPSG_LAMBERT72;
	if (opts.importBoxLL)
	{
		auto [lon0, lat0, lon1, lat1] = *opts.importBoxLL;
		geo::BBox ll{lon0, lat0, lon1, lat1};
		epsg = geo::suggestEpsg({0.5 * (ll.xmin + ll.xmax), 0.5 * (ll.ymin + ll.ymax)});
		box  = geo::transformBox(ll, geo::EPSG_WGS84, epsg);
		box  = {std::floor(box.xmin), std::floor(box.ymin), std::ceil(box.xmax), std::ceil(box.ymax)};
		log::info("working projection: {} (EPSG:{})", geo::crsName(epsg), epsg);
	}
	else if (opts.importBox)
	{
		auto [x0, y0, x1, y1] = *opts.importBox;
		box                   = {x0, y0, x1, y1};
	}
	geo::LocalLibrary library;
	for (auto const &d : opts.libraryDirs)
		library.addFolder(d);
	if (!opts.libraryDirs.empty())
		library.rescan();
	geo::TerrainBuilder builder(library);
	geo::Progress       progress;

	log::info("probe {}x{} m at {} m", box.width(), box.height(), opts.importCell);
	for (auto const &r : builder.probe(box, epsg, progress))
		log::info("  {:<40} footprint {:5.1f} %  data {}{}", r.name, 100.0 * r.footprint,
		          r.data < 0 ? std::string("-") : std::format("{:5.1f} %", 100.0 * r.data), r.error.empty() ? "" : "  (" + r.error + ")");

	geo::ImportRequest req{.box = box, .epsg = epsg, .cell = opts.importCell, .name = opts.importName, .label = opts.importName};
	if (std::isfinite(opts.importLevel))
	{
		req.initial = geo::InitialWater::Level;
		req.level   = opts.importLevel;
	}
	double span  = std::max(box.width(), box.height());
	req.depthRef = span > 5000 ? 5.0 : span > 1000 ? 4.0 : 2.0;
	req.zScale   = span > 5000 ? 5.0 : span > 1000 ? 4.0 : 2.0;
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

	// optional channel carving along a polyline "x1,y1;x2,y2;..."
	if (!opts.carve.empty())
	{
		geo::BedEdit edit{.type = geo::BedEdit::Type::Channel, .width = opts.carveWidth, .depth = opts.carveDepth};
		for (auto [x, y] : opts.carve)
			edit.points.push_back({x, y});
		geo::GridGeometry g{res.bed.nx, res.bed.ny, res.bed.x0, res.bed.y0, res.bed.cell, res.bed.epsg};
		size_t            n = geo::applyBedEdit(res.bed.v, g, edit);
		log::info("carved channel: {} point(s), {} m wide, {} m deep, {} cell(s) modified", edit.points.size(), edit.width, edit.depth, n);
	}

	geo::Image ortho;
	if (opts.importOrtho)
	{
		std::string err, attribution;
		ortho = geo::fetchOrthophoto(box, epsg, 2048, &err, &attribution);
		if (ortho.empty())
			log::warn("orthophoto: {}", err);
		else
		{
			res.attributions.push_back(attribution);
			log::info("orthophoto {}x{} ({})", ortho.width, ortho.height, attribution);
		}
	}
	auto assets = resolveAssetDir(opts.assetDir);
	auto data   = saveImport(res, req, ortho, assets);
	log::info("case '{}' saved in {}", data.name, assets);

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
