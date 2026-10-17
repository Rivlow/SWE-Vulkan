#include "app/Headless.h"

#include "cases/Cases.h"
#include "cases/Npy.h"
#include "solver/Solver.h"
#include "util/Log.h"
#include "vk/Context.h"

#include <nlohmann/json.hpp>

#include <chrono>
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

}        // namespace swe
