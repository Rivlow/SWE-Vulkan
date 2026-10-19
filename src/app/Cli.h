#pragma once
#include "cases/CaseTypes.h"
#include "cases/Cases.h"

#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace swe {

// --water i0,j0,i1,j1,depth
struct WaterBlockOption
{
	CellRect rect;
	float    depth = 0.0f;
};

// --zone kind,value,i0,j0,i1,j1 (value in m DNG for a level zone)
struct ZoneOption
{
	ZoneKind    kind = ZoneKind::Source;
	std::string name;        // as typed: source | sink | level
	double      value = 0.0;
	CellRect    rect;
};

// --bridge i0,j0,i1,j1,z_soffit,deck_thickness (m, soffit in the case's datum: m DNG for imported cases)
struct BridgeOption
{
	CellRect rect;
	double   zRoof = 0.0, thickness = 1.0;
};

// --rain / --infiltration i0,j0,i1,j1,mm_per_h: a chronology zone with a constant rate from t = 0
struct ChronologyZoneOption
{
	bool     rain = true;
	CellRect rect;
	double   mmPerHour = 0.0;
};

using Box = std::array<double, 4>;        // xmin, ymin, xmax, ymax

// Command-line options. Every flag is optional; see printUsage() in main.cpp.
struct Options
{
	bool        headless       = false;
	bool        selftest       = false;
	bool        validation     = true;         // Debug default on, Release default off
	bool        syncValidation = false;        // --sync-validation: synchronization hazards (slow)
	int         gpu            = -1;           // --gpu <index>: force a physical device
	std::string caseName       = "dam_break";  // --case
	bool        caseExplicit   = false;        // --case given (else the last case of swe_settings.json)
	int         colorMode      = -1;           // --color 0..10 (ColorMode)
	std::string scheme;                        // --scheme (empty = case default)
	int         drainPasses    = -1;           // --drain <n> Jacobi passes of the drain limiter (0 = off; -1 = case default)
	int         steps          = 0;            // --steps (headless)
	double      untilTime      = 0.0;          // --time <s> (headless): run until the simulated time reaches it
	std::string dumpDir;                       // --dump <dir> (headless)
	std::string assetDir;                      // --assets <dir>
	std::string capturePath;                   // --capture <png>: screenshot after --capture-frames frames, then quit
	int         captureFrames  = 120;
	int         stepsPerFrame  = -1;           // --steps-per-frame (window)
	bool        showMesh       = false;        // --mesh
	bool        blocks         = false;        // --blocks: raw cells (flat-topped columns)
	std::string recordPath;                    // --record <file.swerec>: record h, qx, qy (window and headless)
	std::string recordDir;                     // --record-dir <dir>: folder of the records (default: records/ next to the exe)
	double      recordEvery    = 1.0;          // --record-every <s>: simulated seconds between records
	bool        recordHalf     = false;        // --record-f16: 16-bit cells (h in mm, float16 q) instead of float32
	float       gwContours     = 0.0f;         // --gw-contours <step>: water table contour lines (m), 0 = off
	bool        gwXray         = false;        // --gw-xray: water table seen through the ground
	bool        gwLines        = false;        // --gw-lines: streamlines of the aquifer
	bool        showProfile    = false;        // --show-profile: open the profile tab of the plots
	int         stage          = -1;           // --stage carte|terrain|conditions|simulation (App::Stage), -1 = default
	bool        run            = false;        // --run: start in the Simulation stage, not paused (captures, scripts)
	std::optional<CellRect> crop;              // --crop i0,j0,i1,j1: cut box
	std::vector<std::pair<int, int>> probes;   // --probe i,j (repeatable)
	// geographic import (headless): --import xmin,ymin,xmax,ymax (Lambert 72) --cell m --library <dir>
	std::optional<Box> importBox;          // Lambert 72 metres
	std::optional<Box> importBoxLL;        // --import-ll lonmin,latmin,lonmax,latmax (WGS 84): projection chosen automatically
	double      importCell = 5.0;
	std::vector<std::string> libraryDirs;
	std::string importName = "import";
	bool        importOrtho = false;        // --ortho: also fetch the SPW orthophoto
	double      importLevel = std::numeric_limits<double>::quiet_NaN();        // --level <m DNG>: still water
	bool        hideGui     = false;        // --no-gui: 3D view only (F11 toggles)
	std::vector<std::pair<double, double>> carve;        // --carve "x1,y1;x2,y2;..." (Lambert 72) with --carve-width/--carve-depth
	double      carveWidth = 20.0, carveDepth = 2.0;
	// scripted edits of the loaded case (window): --water i0,j0,i1,j1,depth ; --zone kind,value,i0,j0,i1,j1 ; --top
	std::vector<WaterBlockOption> waterBlocks;
	std::vector<ZoneOption>       zones;
	std::vector<ChronologyZoneOption> chronologyZones;        // --rain / --infiltration
	std::vector<BridgeOption>     bridges;                    // --bridge
	bool                          topView = false;
	bool        allTiles       = false;        // --all-tiles: compute dry tiles too (reference for the active tiles)
	int         profileSteps   = 0;            // --profile <n>: GPU time per phase over n more steps (headless)
	float       zScale         = -1.0f;        // --z-scale (default: case value)
	bool        help           = false;

	bool        legacyLimiter  = false;        // --legacy-limiter: Venkatakrishnan as in the Python reference (validation)
	double      hEps = -1.0, hDry = -1.0;      // --h-eps / --h-dry <m>: dry and thin-film thresholds (default: case)
	bool        reference = false;             // --reference: the Python reference's numerics (tools/validate.py)
	CaseOverrides caseOverrides() const
	{
		return {.scheme = scheme, .drainPasses = drainPasses, .legacyLimiter = legacyLimiter, .reference = reference,
		        .hEps = hEps, .hDry = hDry};
	}
};

// "1.5,2,3" -> {1.5, 2, 3}; throws unless exactly `count` numbers (count = 0: any)
inline std::vector<double> parseNumbers(std::string_view text, size_t count, char const *option, char sep = ',')
{
	std::vector<double> v;
	size_t              pos = 0;
	try
	{
		while (pos <= text.size())
		{
			size_t end = text.find(sep, pos);
			auto   tok = std::string(text.substr(pos, end == std::string_view::npos ? std::string_view::npos : end - pos));
			size_t used = 0;
			v.push_back(std::stod(tok, &used));
			if (used != tok.size())
				throw std::invalid_argument(tok);
			if (end == std::string_view::npos)
				break;
			pos = end + 1;
		}
	}
	catch (std::exception const &)
	{
		v.clear();
	}
	if (v.empty() || (count != 0 && v.size() != count))
		throw std::runtime_error(std::string(option) + ": cannot read '" + std::string(text) + "'");
	return v;
}

inline CellRect parseRect(std::vector<double> const &v, size_t first)
{
	return {uint32_t(v[first]), uint32_t(v[first + 1]), uint32_t(v[first + 2]), uint32_t(v[first + 3])};
}

inline Box parseBox(std::string_view text, char const *option)
{
	auto v = parseNumbers(text, 4, option);
	Box  b{v[0], v[1], v[2], v[3]};
	if (!(b[2] > b[0] && b[3] > b[1]))
		throw std::runtime_error(std::string(option) + ": empty box");
	return b;
}

inline Options parseOptions(int argc, char **argv)
{
	Options o;
#ifdef NDEBUG
	o.validation = false;
#endif
	auto next = [&](int &i) -> std::string {
		if (i + 1 >= argc)
			throw std::runtime_error(std::string("missing value after ") + argv[i]);
		return argv[++i];
	};
	for (int i = 1; i < argc; ++i)
	{
		std::string_view a = argv[i];
		if (a == "--headless")               o.headless = true;
		else if (a == "--selftest")          o.selftest = true;
		else if (a == "--validation")        o.validation = true;
		else if (a == "--no-validation")     o.validation = false;
		else if (a == "--sync-validation")   o.syncValidation = o.validation = true;
		else if (a == "--gpu")               o.gpu = std::stoi(next(i));
		else if (a == "--case")              { o.caseName = next(i); o.caseExplicit = true; }
		else if (a == "--color")             o.colorMode = std::stoi(next(i));
		else if (a == "--scheme")            o.scheme = next(i);
		else if (a == "--drain")             o.drainPasses = std::stoi(next(i));
		else if (a == "--legacy-limiter")    o.legacyLimiter = true;
		else if (a == "--reference")         o.reference = true;
		else if (a == "--h-eps")             o.hEps = std::stod(next(i));
		else if (a == "--h-dry")             o.hDry = std::stod(next(i));
		else if (a == "--steps")             o.steps = std::stoi(next(i));
		else if (a == "--time")              o.untilTime = std::stod(next(i));
		else if (a == "--dump")              o.dumpDir = next(i);
		else if (a == "--assets")            o.assetDir = next(i);
		else if (a == "--capture")           o.capturePath = next(i);
		else if (a == "--capture-frames")    o.captureFrames = std::stoi(next(i));
		else if (a == "--steps-per-frame")   o.stepsPerFrame = std::stoi(next(i));
		else if (a == "--mesh")              o.showMesh = true;
		else if (a == "--blocks")            o.blocks = true;
		else if (a == "--record")            o.recordPath = next(i);
		else if (a == "--record-dir")        o.recordDir = next(i);
		else if (a == "--record-every")      o.recordEvery = std::stod(next(i));
		else if (a == "--record-f16")        o.recordHalf = true;
		else if (a == "--gw-contours")       o.gwContours = std::stof(next(i));
		else if (a == "--gw-xray")           o.gwXray = true;
		else if (a == "--gw-lines")          o.gwLines = true;
		else if (a == "--show-profile")      o.showProfile = true;
		else if (a == "--run")               o.run = true;
		else if (a == "--stage")
		{
			std::string s = next(i);
			if (s == "carte" || s == "map")                  o.stage = 0;
			else if (s == "terrain")                        o.stage = 1;
			else if (s == "conditions")                     o.stage = 2;
			else if (s == "simulation")                     o.stage = 3;
			else throw std::runtime_error("--stage carte | terrain | conditions | simulation");
		}
		else if (a == "--crop")              o.crop = parseRect(parseNumbers(next(i), 4, "--crop i0,j0,i1,j1"), 0);
		else if (a == "--import")            o.importBox = parseBox(next(i), "--import xmin,ymin,xmax,ymax");
		else if (a == "--import-ll")         o.importBoxLL = parseBox(next(i), "--import-ll lonmin,latmin,lonmax,latmax");
		else if (a == "--cell")              o.importCell = std::stod(next(i));
		else if (a == "--library")           o.libraryDirs.push_back(next(i));
		else if (a == "--name")              o.importName = next(i);
		else if (a == "--ortho")             o.importOrtho = true;
		else if (a == "--level")             o.importLevel = std::stod(next(i));
		else if (a == "--no-gui")            o.hideGui = true;
		else if (a == "--carve")
		{
			std::string s = next(i);
			for (size_t pos = 0; pos < s.size();)
			{
				size_t end = s.find(';', pos);
				auto   v   = parseNumbers(std::string_view(s).substr(pos, end == std::string::npos ? std::string::npos : end - pos), 2,
				                          "--carve x1,y1;x2,y2;...");
				o.carve.emplace_back(v[0], v[1]);
				if (end == std::string::npos)
					break;
				pos = end + 1;
			}
		}
		else if (a == "--carve-width")       o.carveWidth = std::stod(next(i));
		else if (a == "--carve-depth")       o.carveDepth = std::stod(next(i));
		else if (a == "--water")
		{
			auto v = parseNumbers(next(i), 5, "--water i0,j0,i1,j1,depth");
			o.waterBlocks.push_back({parseRect(v, 0), float(v[4])});
		}
		else if (a == "--zone")
		{
			// kind,value,i0,j0,i1,j1
			std::string s     = next(i);
			auto        comma = s.find(',');
			ZoneOption  z;
			z.name = s.substr(0, comma);
			if (z.name == "source")     z.kind = ZoneKind::Source;
			else if (z.name == "sink")  z.kind = ZoneKind::Sink;
			else if (z.name == "level") z.kind = ZoneKind::Level;
			else throw std::runtime_error("--zone kind,value,i0,j0,i1,j1: kind = source | sink | level");
			auto v  = parseNumbers(comma == std::string::npos ? "" : std::string_view(s).substr(comma + 1), 5, "--zone kind,value,i0,j0,i1,j1");
			z.value = v[0];
			z.rect  = parseRect(v, 1);
			o.zones.push_back(z);
		}
		else if (a == "--rain" || a == "--infiltration")
		{
			auto v = parseNumbers(next(i), 5, "--rain / --infiltration i0,j0,i1,j1,mm_per_h");
			o.chronologyZones.push_back({a == "--rain", parseRect(v, 0), v[4]});
		}
		else if (a == "--bridge")
		{
			auto v = parseNumbers(next(i), 6, "--bridge i0,j0,i1,j1,z_soffit,thickness");
			o.bridges.push_back({parseRect(v, 0), v[4], v[5]});
		}
		else if (a == "--top")               o.topView = true;
		else if (a == "--all-tiles")         o.allTiles = true;
		else if (a == "--profile")           o.profileSteps = std::stoi(next(i));
		else if (a == "--probe")
		{
			auto v     = next(i);
			auto comma = v.find(',');
			if (comma == std::string::npos)
				throw std::runtime_error("--probe expects i,j");
			o.probes.emplace_back(std::stoi(v.substr(0, comma)), std::stoi(v.substr(comma + 1)));
		}
		else if (a == "--z-scale")           o.zScale = std::stof(next(i));
		else if (a == "-h" || a == "--help") o.help = true;
		else throw std::runtime_error("unknown option: " + std::string(a));
	}
	return o;
}

}        // namespace swe
