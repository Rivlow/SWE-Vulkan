#pragma once
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace swe {

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
	int         colorMode      = -1;           // --color 0..4 (realistic, depth, speed, Froude, elevation)
	std::string scheme;                        // --scheme (empty = case default)
	int         steps          = 0;            // --steps (headless)
	std::string dumpDir;                       // --dump <dir> (headless)
	std::string assetDir;                      // --assets <dir>
	std::string capturePath;                   // --capture <png>: screenshot after --capture-frames frames, then quit
	int         captureFrames  = 120;
	int         stepsPerFrame  = -1;           // --steps-per-frame (window)
	bool        showMesh       = false;        // --mesh
	std::vector<std::pair<int, int>> probes;   // --probe i,j (repeatable)
	// geographic import (headless): --import xmin,ymin,xmax,ymax (Lambert 72) --cell m --library <dir>
	std::string importBox;
	double      importCell = 5.0;
	std::vector<std::string> libraryDirs;
	std::string importName = "import";
	bool        importOrtho = false;        // --ortho: also fetch the SPW orthophoto
	double      importLevel = std::numeric_limits<double>::quiet_NaN();        // --level <m DNG>: still water
	bool        hideGui     = false;        // --no-gui: 3D view only (F11 toggles)
	std::string carve;                      // --carve "x1,y1;x2,y2;..." (Lambert 72) with --carve-width/--carve-depth
	double      carveWidth = 20.0, carveDepth = 2.0;
	// scripted edits of the loaded case (window): --water i0,j0,i1,j1,depth ; --zone kind,value,i0,j0,i1,j1 ; --top
	std::vector<std::string> waterBlocks, zoneDefs;
	bool                     topView = false;
	float       zScale         = -1.0f;        // --z-scale (default: case value)
	bool        help           = false;
};

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
		else if (a == "--steps")             o.steps = std::stoi(next(i));
		else if (a == "--dump")              o.dumpDir = next(i);
		else if (a == "--assets")            o.assetDir = next(i);
		else if (a == "--capture")           o.capturePath = next(i);
		else if (a == "--capture-frames")    o.captureFrames = std::stoi(next(i));
		else if (a == "--steps-per-frame")   o.stepsPerFrame = std::stoi(next(i));
		else if (a == "--mesh")              o.showMesh = true;
		else if (a == "--import")            o.importBox = next(i);
		else if (a == "--cell")              o.importCell = std::stod(next(i));
		else if (a == "--library")           o.libraryDirs.push_back(next(i));
		else if (a == "--name")              o.importName = next(i);
		else if (a == "--ortho")             o.importOrtho = true;
		else if (a == "--level")             o.importLevel = std::stod(next(i));
		else if (a == "--no-gui")            o.hideGui = true;
		else if (a == "--carve")             o.carve = next(i);
		else if (a == "--carve-width")       o.carveWidth = std::stod(next(i));
		else if (a == "--carve-depth")       o.carveDepth = std::stod(next(i));
		else if (a == "--water")             o.waterBlocks.push_back(next(i));
		else if (a == "--zone")              o.zoneDefs.push_back(next(i));
		else if (a == "--top")               o.topView = true;
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
