#include "app/App.h"
#include "app/Headless.h"
#include "app/SelfTest.h"
#include "geo/Crs.h"
#include "app/Cli.h"
#include "util/Log.h"
#include "vk/Context.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

void printUsage()
{
	std::cout << R"(SWE-Vulkan - Saint-Venant 2D solver and viewer

usage: swe [options]
  --case <name>        case to load (dam_break, meuse, vesdre)       [dam_break]
  --scheme <name>      time scheme (euler, ssprk2, ssprk3)
  --drain <n>          drain limiter: Jacobi passes per stage, 0 = off (clamp h < 0)   [case, 4]
  --gpu <index>        force a physical device (see the GPU list printed at start)
  --assets <dir>       asset directory (cases/, data/)
  --validation / --no-validation   Vulkan validation layers (default: on in Debug)
  --sync-validation    also enable synchronization validation (slow)
  --steps-per-frame <n>  solver steps per displayed frame                [30]
  --color <0..10>      colouring: realistic, depth, speed, Froude, elevation; groundwater cases (on the
                       ground): table depth, saturation, exchange, infiltrated depth, head; 10 = rain
  --mesh               cell boundaries overlay
  --blocks             raw cells: flat-topped columns, no interpolation
  --record <file.swerec>   record h, qx, qy every --record-every simulated seconds (tools/swerec.py reads it;
                       window and --headless runs)
  --record-dir <dir>   folder of the records started from the GUI (default: records/ next to the executable,
                       kept in swe_settings.json); with --headless and no --record: <dir>/<case>_<date>.swerec
  --record-every <s>                                                     [1.0]
  --record-f16         16-bit cells (h to the mm, q as float16) instead of exact float32: half the bytes
  --gw-contours <m>    water table contour lines with this step (groundwater cases)
  --gw-xray            water table seen through the ground; --gw-lines: streamlines of the aquifer
  --crop i0,j0,i1,j1   cut box: only these cells are drawn, the walls show the soil column
  --show-profile       open the vertical profile in the plots (with --probe i,j twice: probe 1 -> 2)
  --stage <s>          start at this stage: carte (default), terrain (default with --case), conditions, simulation
  --run                start simulating at once (Simulation stage, not paused); otherwise everything starts paused
  --z-scale <f>        vertical exaggeration (default: case value)
  --capture <png>      save a screenshot after --capture-frames frames, then quit
  --capture-frames <n>                                                   [120]
  --headless           no window (batch mode)
    --selftest         run the compute self-test and exit
    --steps <n>        number of solver steps
    --time <s>         run until the simulated time reaches s (after --steps, if any)
    --dump <dir>       write Q.npy, zb.npy and meta.json after the run
    --profile <n>      then GPU time of each phase of a step, averaged over n more steps
  --import xmin,ymin,xmax,ymax   geographic import (Lambert 72 m) of the bed, saved as a case
  --import-ll lonmin,latmin,lonmax,latmax   same with a WGS 84 box (projection chosen automatically)
    --cell <m>         target cell size                                          [5]
    --library <dir>    folder of local GeoTIFF (SPW DTM / bathymetry), repeatable
    --name <n>         case name                                                 [import]
    --dump <dir>       also write <name>.tif and zb_abs.npy
    --ortho            also fetch the SPW orthophoto for the drape
    --level <m>        still water at this elevation (m DNG) as initial state
  --no-gui             window without panels (F11 toggles)
  --top                start with the top view
  --water i0,j0,i1,j1,h   water block (depth h) on a cell rectangle, repeatable
  --zone kind,value,i0,j0,i1,j1   per-cell zone (source m3/s | sink | level m DNG), repeatable
  --bridge i0,j0,i1,j1,z,e        bridge deck over a cell rectangle: soffit at z (m, case datum), deck thickness e;
                       pressurised flow above the soffit, overtopping above the deck (repeatable)
  --rain i0,j0,i1,j1,mm_h         rain zone, constant rate from t = 0 (repeatable; --infiltration: same,
                       signed infiltration chronology zone)
                       (--water and --zone also apply to --headless runs)
  --all-tiles          compute dry tiles too (reference for the active-tile scheme; same results, slower)
)";
}

}        // namespace

int main(int argc, char **argv)
{
#ifdef _WIN32
	// Windows application (no console of its own): when started from a console, log there; when the standard
	// handles are already redirected (tools, pipes), keep them
	if (GetStdHandle(STD_OUTPUT_HANDLE) == nullptr && AttachConsole(ATTACH_PARENT_PROCESS))
	{
		FILE *f = nullptr;
		freopen_s(&f, "CONOUT$", "w", stdout);
		freopen_s(&f, "CONOUT$", "w", stderr);
	}
#endif
	try
	{
		auto opts = swe::parseOptions(argc, argv);
		if (opts.help)
		{
			printUsage();
			return EXIT_SUCCESS;
		}

		if (opts.importBox || opts.importBoxLL)
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
