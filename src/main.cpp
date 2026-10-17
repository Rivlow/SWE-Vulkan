#include "app/App.h"
#include "app/Headless.h"
#include "app/SelfTest.h"
#include "geo/Crs.h"
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
  --steps-per-frame <n>  solver steps per displayed frame                [30]
  --color <0..4>       water colouring: realistic, depth, speed, Froude, elevation
  --mesh               wireframe overlay
  --z-scale <f>        vertical exaggeration (default: case value)
  --capture <png>      save a screenshot after --capture-frames frames, then quit
  --capture-frames <n>                                                   [120]
  --headless           no window (batch mode)
    --selftest         run the compute self-test and exit
    --steps <n>        number of solver steps
    --dump <dir>       write Q.npy, zb.npy and meta.json after the run
  --import xmin,ymin,xmax,ymax   geographic import (Lambert 72 m) of the bed, saved as a case
    --cell <m>         target cell size                                          [5]
    --library <dir>    folder of local GeoTIFF (SPW DTM / bathymetry), repeatable
    --name <n>         case name                                                 [import]
    --dump <dir>       also write <name>.tif and zb_abs.npy
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

		if (!opts.importBox.empty())
		{
			int code = swe::runImport(opts);
			swe::geo::shutdown();
			std::cout.flush();
			return code;
		}

		if (opts.headless)
		{
			swe::vkc::Context ctx({.validation = opts.validation, .syncValidation = opts.syncValidation, .forcedGpu = opts.gpu});
			if (opts.selftest)
				return swe::runSelfTest(ctx) ? EXIT_SUCCESS : EXIT_FAILURE;
			return swe::runHeadless(opts, ctx);
		}

		{
			swe::App app(opts);
			app.run();
		}
		swe::geo::shutdown();
		return EXIT_SUCCESS;
	}
	catch (std::exception const &e)
	{
		swe::log::error("{}", e.what());
		return EXIT_FAILURE;
	}
}
