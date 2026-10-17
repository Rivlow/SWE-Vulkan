#include "app/App.h"
#include "app/SelfTest.h"
#include "util/Cli.h"
#include "util/Log.h"
#include "vk/Context.h"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

void printUsage()
{
	std::cout << R"(SWE-Vulkan - Saint-Venant 2D solver and viewer

usage: swe [options]
  --case <name>        case to load (dam_break, meuse, vesdre)       [dam_break]
  --scheme <name>      time scheme (euler, ssprk2, ssprk3, rk4, dopri5)
  --gpu <index>        force a physical device (see the GPU list printed at start)
  --assets <dir>       asset directory (cases/, data/)
  --validation / --no-validation   Vulkan validation layers (default: on in Debug)
  --sync-validation    also enable synchronization validation (slow)
  --headless           no window (batch mode)
    --selftest         run the compute self-test and exit
    --steps <n>        number of solver steps
    --dump <dir>       write Q.npy, zb.npy and meta.json after the run
)";
}

}        // namespace

int main(int argc, char **argv)
{
	try
	{
		auto opts = swe::parseOptions(argc, argv);
		if (opts.help)
		{
			printUsage();
			return EXIT_SUCCESS;
		}

		if (opts.headless)
		{
			swe::vkc::Context ctx({.validation = opts.validation, .syncValidation = opts.syncValidation, .forcedGpu = opts.gpu});
			if (opts.selftest)
				return swe::runSelfTest(ctx) ? EXIT_SUCCESS : EXIT_FAILURE;
			swe::log::error("headless mode needs --selftest (solver runs arrive with milestone M1)");
			return EXIT_FAILURE;
		}

		swe::App app(opts);
		app.run();
		return EXIT_SUCCESS;
	}
	catch (std::exception const &e)
	{
		swe::log::error("{}", e.what());
		return EXIT_FAILURE;
	}
}
