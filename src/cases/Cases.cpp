#include "cases/Cases.h"

#include "cases/GridOps.h"
#include "cases/Npy.h"
#include "util/Log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

namespace swe {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

json readJson(fs::path const &path)
{
	std::ifstream f(path);
	if (!f)
		throw std::runtime_error("cannot open " + path.string());
	return json::parse(f, nullptr, true, /*ignore_comments=*/true);
}

// Physics, numerics, boundary conditions and rendering hints shared by every case type.
void readCommon(CaseData &c, json const &j)
{
	c.label     = j.value("label", c.name);
	c.manning_n = j.value("manning_n", 0.0);
	c.cfl       = j.value("cfl", 0.4);
	c.scheme    = j.value("scheme", std::string("ssprk2"));
	if (j.contains("render"))
	{
		c.depth_ref = j["render"].value("depth_ref", 1.0);
		c.z_scale_0 = j["render"].value("z_scale_0", 2.0);
	}
}

// "bc": {"west": ["wall"], "south": ["inflow", 600], "north": ["level", 58.5 | "eta_0"]}
void readBoundaries(CaseData &c, json const &j)
{
	static char const *sides[4] = {"west", "east", "south", "north"};
	for (int s = 0; s < 4; ++s)
	{
		c.bc[s] = {BcKind::Wall, 0.0};
		if (!j.contains("bc") || !j["bc"].contains(sides[s]))
			continue;
		auto const &b    = j["bc"][sides[s]];
		auto        kind = b.at(0).get<std::string>();
		if (kind == "wall")
			c.bc[s].kind = BcKind::Wall;
		else if (kind == "inflow")
			c.bc[s] = {BcKind::Inflow, b.at(1).get<double>()};
		else if (kind == "level")
		{
			c.bc[s].kind  = BcKind::Level;
			c.bc[s].value = b.at(1).is_string() && b.at(1) == "eta_0" ? c.eta_0 : b.at(1).get<double>();
		}
		else
			throw std::runtime_error("unknown boundary condition '" + kind + "' (" + c.name + ")");
	}
}

void allocState(CaseData &c)
{
	c.zb.assign(c.cells(), 0.0f);
	c.h0.assign(c.cells(), 0.0f);
	c.hu0.assign(c.cells(), 0.0f);
	c.hv0.assign(c.cells(), 0.0f);
}

// Port of cases/dam_break.py: free surface eta_0 behind the dam at x = Lx/2, dry bed downstream, with a bump
// in the bed. f32 with the same operation order as the Taichi kernel. lake_at_rest: still water everywhere.
void buildBump(CaseData &c, json const &j, bool damBreak)
{
	c.nx    = j.value("nx", 250u);
	c.ny    = j.value("ny", 250u);
	c.Lx    = j.value("Lx", 10.0);
	c.Ly    = j.value("Ly", 10.0);
	c.eta_0 = j.value("eta_0", 1.0);
	allocState(c);

	float const dx   = static_cast<float>(c.dx());
	float const dy   = static_cast<float>(c.dy());
	float const half = static_cast<float>(c.Lx / 2);
	float const eta0 = static_cast<float>(c.eta_0);
	for (uint32_t i = 0; i < c.nx; ++i)
		for (uint32_t jj = 0; jj < c.ny; ++jj)
		{
			float x  = (static_cast<float>(i) + 0.5f) * dx;        // cell centre
			float y  = (static_cast<float>(jj) + 0.5f) * dy;
			float ex = x - 7.5f, ey = y - 5.0f;
			float z  = 0.6f * std::exp(-(ex * ex + ey * ey) / 0.8f);
			auto  k  = size_t(i) * c.ny + jj;
			c.zb[k]  = z;
			if (!damBreak || x < half)
				c.h0[k] = std::max(eta0 - z, 0.0f);
		}
}

// Port of the "meuse" branch of main.py + cases/river.py: river reach from a real bathymetry, at rest at
// eta_0; the flow is then driven by the boundary conditions (upstream discharge, downstream level).
void buildRiver(CaseData &c, json const &j, fs::path const &assets)
{
	auto   a    = npy::load(assets / j.at("bathymetry").get<std::string>());
	double cell = j.at("cell").get<double>();
	double etaA = j.at("eta_abs").get<double>();
	double wall = j.at("wall_height").get<double>();
	if (a.shape.size() != 2)
		throw std::runtime_error("river: bathymetry must be a 2-D array");
	auto zRaw = a.asFloat();

	c.nx    = static_cast<uint32_t>(a.shape[0]);
	c.ny    = static_cast<uint32_t>(a.shape[1]);
	c.Lx    = c.nx * cell;
	c.Ly    = c.ny * cell;
	c.z_ref = std::floor(double(grid::nanMin(zRaw)));        // relative elevations: better f32 precision
	c.eta_0 = etaA - c.z_ref;
	allocState(c);

	// outside the survey (banks, quays, bridge piers): walls at eta + wall_height; float32 like numpy
	float const wallZ = static_cast<float>(etaA + wall);
	float const zRef  = static_cast<float>(c.z_ref);
	float const eta0  = static_cast<float>(c.eta_0);
	for (size_t k = 0; k < c.cells(); ++k)
	{
		c.zb[k] = (std::isnan(zRaw[k]) ? wallZ : zRaw[k]) - zRef;
		c.h0[k] = std::max(eta0 - c.zb[k], 0.0f);
	}
}

// Port of the "vesdre" branch of main.py + cases/lake.py: DTM clipped to sub-basins, refined, with an
// initial lake in one sub-basin and an outlet at the lowest cell of the basin.
void buildDtmLake(CaseData &c, json const &j, fs::path const &assets)
{
	auto     npz       = npy::loadNpz(assets / j.at("dtm").get<std::string>());
	auto     r         = j.value("refine", 1u);
	auto     lakeBasin = j.at("lake_basin").get<int>();
	double   lakeLevel = j.at("lake_level").get<double>();
	float    wall      = static_cast<float>(j.at("wall_height").get<double>());
	auto    &zA        = npz.at("z");
	auto    &bA        = npz.at("basin");
	uint32_t nx0 = static_cast<uint32_t>(zA.shape.at(0)), ny0 = static_cast<uint32_t>(zA.shape.at(1));
	double   cell0 = npz.at("cell").asDouble().at(0);

	auto z0     = zA.asFloat();
	auto basin0 = bA.asDouble();
	std::vector<uint8_t> nan0(z0.size());
	std::vector<int>     basinI(z0.size());
	for (size_t k = 0; k < z0.size(); ++k)
	{
		nan0[k]   = std::isnan(z0[k]) ? 1 : 0;
		basinI[k] = static_cast<int>(basin0[k]);
	}

	c.nx = nx0 * r;
	c.ny = ny0 * r;
	double cell = cell0 / r;
	c.Lx        = c.nx * cell;
	c.Ly        = c.ny * cell;

	c.outside  = grid::refineNearest(nan0, nx0, ny0, r);
	auto basin = grid::refineNearest(basinI, nx0, ny0, r);
	auto zFill = grid::refineBilinear(grid::fillOutside(z0, nx0, ny0, wall), nx0, ny0, r);

	std::vector<float> zRaw(zFill.size());        // elevations, NaN outside the basin
	for (size_t k = 0; k < zRaw.size(); ++k)
		zRaw[k] = c.outside[k] ? kNaN : zFill[k];
	c.z_ref = std::floor(double(grid::nanMin(zRaw)));
	c.eta_0 = lakeLevel - c.z_ref;
	allocState(c);

	float const zRef = static_cast<float>(c.z_ref);
	float const lake = static_cast<float>(lakeLevel);
	for (size_t k = 0; k < c.cells(); ++k)
	{
		c.zb[k] = zFill[k] - zRef;
		if (basin[k] == lakeBasin && zRaw[k] < lake)        // NaN compares false
			c.h0[k] = lake - zRaw[k];
	}

	// outlet: lowest cell of the basin and its neighbours inside the basin (~ +-refine cells)
	if (j.value("outlet", true))
	{
		size_t kmin = grid::nanArgMin(zRaw);
		int    io = int(kmin / c.ny), jo = int(kmin % c.ny), ir = int(r);
		c.sink.assign(c.cells(), 0u);
		for (int i = std::max(io - ir, 0); i <= std::min(io + ir, int(c.nx) - 1); ++i)
			for (int jj = std::max(jo - ir, 0); jj <= std::min(jo + ir, int(c.ny) - 1); ++jj)
			{
				size_t k = size_t(i) * c.ny + jj;
				c.sink[k] = std::isnan(zRaw[k]) ? 0u : 1u;
			}
	}
}

// Case saved by the geographic import: everything is in the npz.
void buildImported(CaseData &c, json const &j, fs::path const &assets)
{
	auto npz = npy::loadNpz(assets / j.at("data").get<std::string>());
	auto &zb = npz.at("zb");
	if (zb.shape.size() != 2)
		throw std::runtime_error("imported: zb must be 2-D");
	c.nx    = static_cast<uint32_t>(zb.shape[0]);
	c.ny    = static_cast<uint32_t>(zb.shape[1]);
	double cell = j.at("cell").get<double>();
	c.Lx    = c.nx * cell;
	c.Ly    = c.ny * cell;
	c.z_ref = j.value("z_ref", 0.0);
	c.eta_0 = j.value("eta_0", 1.0);
	c.zb    = zb.asFloat();
	c.h0    = npz.count("h0") ? npz.at("h0").asFloat() : std::vector<float>(c.cells(), 0.0f);
	c.hu0.assign(c.cells(), 0.0f);
	c.hv0.assign(c.cells(), 0.0f);
	if (npz.count("outside"))
	{
		auto o = npz.at("outside").asFloat();
		c.outside.resize(o.size());
		for (size_t k = 0; k < o.size(); ++k)
			c.outside[k] = o[k] != 0.0f ? 1 : 0;
		if (std::ranges::none_of(c.outside, [](uint8_t v) { return v; }))
			c.outside.clear();
	}
	if (c.h0.size() != c.cells() || (!c.outside.empty() && c.outside.size() != c.cells()))
		throw std::runtime_error("imported: inconsistent array sizes");
}

}        // namespace

GeoOrigin caseOrigin(std::string const &name, std::string const &assetDir)
{
	GeoOrigin g;
	auto      path = fs::path(assetDir) / "cases" / (name + ".json");
	if (!fs::exists(path))
		return g;
	try
	{
		json j = readJson(path);
		if (j.contains("origin") && j.contains("cell"))
		{
			g.x0    = j["origin"].at(0).get<double>();
			g.y0    = j["origin"].at(1).get<double>();
			g.cell  = j["cell"].get<double>();
			g.valid = true;
		}
		g.orthoFile = j.value("orthophoto", std::string{});
	}
	catch (std::exception const &)
	{
	}
	return g;
}

void saveImportedCase(CaseData const &c, ImportMetadata const &meta, std::string const &assetDir)
{
	fs::path assets(assetDir);
	fs::create_directories(assets / "cases");
	fs::create_directories(assets / "data");

	std::map<std::string, npy::Array> arrays;
	arrays["zb"] = npy::makeArray(c.zb, {c.nx, c.ny});
	arrays["h0"] = npy::makeArray(c.h0, {c.nx, c.ny});
	if (!c.outside.empty())
		arrays["outside"] = npy::makeArray(c.outside, {c.nx, c.ny});
	npy::saveNpz(assets / "data" / (c.name + ".npz"), arrays);

	static char const *sides[4] = {"west", "east", "south", "north"};
	json               bc;
	for (int s = 0; s < 4; ++s)
	{
		switch (c.bc[s].kind)
		{
		case BcKind::Wall: bc[sides[s]] = {"wall"}; break;
		case BcKind::Inflow: bc[sides[s]] = {"inflow", c.bc[s].value}; break;
		case BcKind::Level: bc[sides[s]] = {"level", c.bc[s].value}; break;
		}
	}
	json j{{"label", c.label},
	       {"type", "imported"},
	       {"data", "data/" + c.name + ".npz"},
	       {"cell", meta.cell},
	       {"origin", {meta.x0, meta.y0}},
	       {"crs", "EPSG:31370"},
	       {"vertical_datum", "DNG (approx. for foreign sources)"},
	       {"z_ref", c.z_ref},
	       {"eta_0", c.eta_0},
	       {"manning_n", c.manning_n},
	       {"cfl", c.cfl},
	       {"scheme", c.scheme},
	       {"bc", bc},
	       {"render", {{"depth_ref", c.depth_ref}, {"z_scale_0", c.z_scale_0}}},
	       {"sources", meta.sources},
	       {"attributions", meta.attributions}};
	if (!meta.orthoFile.empty())
		j["orthophoto"] = meta.orthoFile;
	std::ofstream(assets / "cases" / (c.name + ".json")) << j.dump(2) << '\n';
}

void updateImportedCase(CaseData const &c, std::string const &assetDir)
{
	fs::path assets(assetDir);
	auto     path = assets / "cases" / (c.name + ".json");
	json     j    = fs::exists(path) ? readJson(path) : json::object();
	if (j.value("type", "") != "imported")
		throw std::runtime_error("only imported cases can be saved in place");

	std::map<std::string, npy::Array> arrays;
	arrays["zb"] = npy::makeArray(c.zb, {c.nx, c.ny});
	arrays["h0"] = npy::makeArray(c.h0, {c.nx, c.ny});
	if (!c.outside.empty())
		arrays["outside"] = npy::makeArray(c.outside, {c.nx, c.ny});
	npy::saveNpz(assets / j.at("data").get<std::string>(), arrays);

	static char const *sides[4] = {"west", "east", "south", "north"};
	json               bc;
	for (int s = 0; s < 4; ++s)
	{
		switch (c.bc[s].kind)
		{
		case BcKind::Wall: bc[sides[s]] = {"wall"}; break;
		case BcKind::Inflow: bc[sides[s]] = {"inflow", c.bc[s].value}; break;
		case BcKind::Level: bc[sides[s]] = {"level", c.bc[s].value}; break;
		}
	}
	j["bc"]        = bc;
	j["manning_n"] = c.manning_n;
	j["cfl"]       = c.cfl;
	j["scheme"]    = c.scheme;
	j["eta_0"]     = c.eta_0;
	j["z_ref"]     = c.z_ref;
	j["edited"]    = true;
	std::ofstream(path) << j.dump(2) << '\n';
}

std::vector<CaseInfo> listCases(std::string const &assetDir)
{
	std::vector<CaseInfo> out;
	auto                  dir = fs::path(assetDir) / "cases";
	if (!fs::exists(dir))
		return out;
	for (auto const &e : fs::directory_iterator(dir))
	{
		if (e.path().extension() != ".json")
			continue;
		CaseInfo info{e.path().stem().string(), e.path().stem().string()};
		try
		{
			info.label = readJson(e.path()).value("label", info.name);
		}
		catch (std::exception const &ex)
		{
			log::warn("{}: {}", e.path().string(), ex.what());
		}
		out.push_back(info);
	}
	std::ranges::sort(out, {}, &CaseInfo::name);
	return out;
}

CaseData makeCase(std::string const &name, std::string const &assetDir)
{
	fs::path assets(assetDir);
	auto     path = assets / "cases" / (name + ".json");
	if (!fs::exists(path))
		throw std::runtime_error("unknown case '" + name + "' (no " + path.string() + ")");
	json j = readJson(path);

	CaseData c;
	c.name    = name;
	auto type = j.at("type").get<std::string>();
	readCommon(c, j);
	if (type == "dam_break")
		buildBump(c, j, true);
	else if (type == "lake_at_rest")
		buildBump(c, j, false);
	else if (type == "river")
		buildRiver(c, j, assets);
	else if (type == "dtm_lake")
		buildDtmLake(c, j, assets);
	else if (type == "imported")
		buildImported(c, j, assets);
	else
		throw std::runtime_error("case '" + name + "': unknown type '" + type + "'");
	readBoundaries(c, j);

	for (float z : c.zb)
		if (!std::isfinite(z))
			throw std::runtime_error("case '" + name + "': non-finite bed elevation");
	return c;
}

}        // namespace swe
