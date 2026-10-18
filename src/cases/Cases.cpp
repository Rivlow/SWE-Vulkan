#include "cases/Cases.h"

#include "cases/GridOps.h"
#include "cases/Npy.h"
#include "util/Log.h"
#include "util/Paths.h"

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

void readZones(CaseData &c, json const &j, std::map<std::string, npy::Array> const &npz);
json zonesJson(CaseData const &c);

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
	c.drain_passes = std::clamp(j.value("drain_passes", 4), 0, MAX_DRAIN_PASSES);
	c.legacyInflow = j.value("legacy_inflow", false);
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

	// outlet: lowest cell of the basin and its neighbours inside the basin (~ +-refine cells), as a Sink zone
	if (j.value("outlet", true))
	{
		size_t   kmin = grid::nanArgMin(zRaw);
		int      io = int(kmin / c.ny), jo = int(kmin % c.ny), ir = int(r);
		uint32_t i0 = uint32_t(std::max(io - ir, 0)), i1 = uint32_t(std::min(io + ir, int(c.nx) - 1));
		uint32_t j0 = uint32_t(std::max(jo - ir, 0)), j1 = uint32_t(std::min(jo + ir, int(c.ny) - 1));
		c.zones    = {{.kind = ZoneKind::Sink, .label = "exutoire", .i0 = i0, .j0 = j0, .i1 = i1, .j1 = j1}};
		c.zoneMask.assign(c.cells(), 0);
		for (uint32_t i = i0; i <= i1; ++i)
			for (uint32_t jj = j0; jj <= j1; ++jj)
				c.zoneMask[c.index(i, jj)] = std::isnan(zRaw[c.index(i, jj)]) ? 0 : 1;
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
	if (auto crs = j.value("crs", std::string("EPSG:31370")); crs.rfind("EPSG:", 0) == 0)
		c.epsg = std::stoi(crs.substr(5));
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
	readZones(c, j, npz);
}

// A per-cell field of the "groundwater" block: a number (uniform), a "file.npy" relative to the assets, or the
// name of an array of the case's npz (imported cases). Empty when the key is absent.
std::vector<float> readField(CaseData const &c, json const &g, char const *key, json const &j, fs::path const &assets)
{
	if (!g.contains(key))
		return {};
	auto const &v = g[key];
	if (v.is_number())
		return std::vector<float>(c.cells(), v.get<float>());
	auto               name = v.get<std::string>();
	std::vector<float> a;
	if (name.ends_with(".npy"))
		a = npy::load(assets / name).asFloat();
	else if (j.contains("data"))
		a = npy::loadNpz(assets / j["data"].get<std::string>()).at(name).asFloat();
	else
		throw std::runtime_error(std::format("groundwater.{}: '{}' is neither a .npy file nor a case npz array", key, name));
	if (a.size() != c.cells())
		throw std::runtime_error(std::format("groundwater.{}: {} values for {} cells", key, a.size(), c.cells()));
	return a;
}

// "groundwater": {"kv": 1e-5, "kh": 1e-4, "sy": 0.25, "depth": 10 | "z_sub": ..., "h0": 2 | "head": ... |
// "table_depth": ..., "datum": "wolfgpu"}. Each field: number, "file.npy" or npz array name (see readField).
void readGroundwater(CaseData &c, json const &j, fs::path const &assets)
{
	auto &gw = c.gw;
	gw       = {};
	if (!j.contains("groundwater"))
		return;
	auto const &g = j["groundwater"];
	size_t const n = c.cells();
	gw.enabled = true;
	gw.wolfgpu = g.value("datum", std::string("physical")) == "wolfgpu";
	gw.kv      = readField(c, g, "kv", j, assets);
	gw.kh      = readField(c, g, "kh", j, assets);
	gw.sy      = readField(c, g, "sy", j, assets);
	if (gw.kv.empty())
		gw.kv.assign(n, 0.0f);
	if (gw.kh.empty())
		gw.kh.assign(n, 0.0f);
	if (gw.sy.empty() || gw.wolfgpu)
		gw.sy.assign(n, gw.wolfgpu ? 1.0f : 0.25f);
	for (size_t k = 0; k < n; ++k)
	{
		gw.kv[k] = std::max(gw.kv[k], 0.0f);
		gw.kh[k] = std::max(gw.kh[k], 0.0f);
		gw.sy[k] = std::clamp(gw.sy[k], 1e-3f, 1.0f);
	}

	// substratum: wolfgpu's aquifer lies on the bed itself
	gw.z_sub = readField(c, g, "z_sub", j, assets);
	if (gw.wolfgpu)
		gw.z_sub = c.zb;
	else if (gw.z_sub.empty())
	{
		auto depth = readField(c, g, "depth", j, assets);
		if (depth.empty())
			depth.assign(n, 10.0f);
		gw.z_sub.resize(n);
		for (size_t k = 0; k < n; ++k)
			gw.z_sub[k] = c.zb[k] - std::max(depth[k], 0.0f);
	}

	// initial saturated thickness: given directly, or from a head / a depth of the table below the ground
	gw.h0 = readField(c, g, "h0", j, assets);
	if (gw.h0.empty())
	{
		auto head  = readField(c, g, "head", j, assets);
		auto depth = readField(c, g, "table_depth", j, assets);
		gw.h0.assign(n, 0.0f);
		for (size_t k = 0; k < n; ++k)
			if (!head.empty())
				gw.h0[k] = head[k] - gw.z_sub[k];
			else if (!depth.empty())
				gw.h0[k] = c.zb[k] - depth[k] - gw.z_sub[k];
	}
	for (size_t k = 0; k < n; ++k)
	{
		gw.h0[k] = std::max(gw.h0[k], 0.0f);
		if (!gw.wolfgpu)        // a table above the ground would exfiltrate at once: start at most saturated
			gw.h0[k] = std::min(gw.h0[k], std::max(c.zb[k] - gw.z_sub[k], 0.0f));
	}
}

// "rain" / "infiltration": {"zones": "all" | [[i0, j0, i1, j1], ...] (zone k = rectangle k) | "file.npy" | npz array,
//  "chronology": [[t, [q_zone1, q_zone2, ...]], ...], "lerp": false, "unit": "m3/s" | "mm/h"}
// Same chronology as wolfgpu's add_infiltration / add_rain: m3/s for a whole zone, constant after the last line,
// nothing before the first one. "mm/h" (convenience for rain) is converted to m3/s with the zone areas.
void readChronology(CaseData &c, CaseData::Chronology &chrono, char const *key, json const &j, fs::path const &assets)
{
	chrono = {};
	if (!j.contains(key))
		return;
	auto const &g = j[key];
	size_t const n = c.cells();
	auto const  &z = g.at("zones");
	if (z.is_string() && z.get<std::string>() == "all")
		chrono.zones.assign(n, 1);
	else if (z.is_array())
	{
		chrono.zones.assign(n, 0);
		for (size_t k = 0; k < z.size(); ++k)
		{
			auto const &r = z[k];
			for (uint32_t i = r.at(0).get<uint32_t>(); i <= std::min(r.at(2).get<uint32_t>(), c.nx - 1); ++i)
				for (uint32_t jj = r.at(1).get<uint32_t>(); jj <= std::min(r.at(3).get<uint32_t>(), c.ny - 1); ++jj)
					chrono.zones[c.index(i, jj)] = static_cast<uint16_t>(k + 1);
		}
	}
	else
	{
		json spec = {{"zones", z}};
		auto f    = readField(c, spec, "zones", j, assets);
		chrono.zones.resize(n);
		for (size_t k = 0; k < n; ++k)
			chrono.zones[k] = static_cast<uint16_t>(std::clamp(f[k], 0.0f, 65535.0f));
	}
	chrono.lerp = g.value("lerp", false);
	for (auto const &line : g.at("chronology"))
	{
		chrono.t.push_back(line.at(0).get<double>());
		chrono.q.push_back(line.at(1).get<std::vector<double>>());
		if (chrono.q.back().size() != chrono.q.front().size())
			throw std::runtime_error(std::format("{}: every chronology line needs the same number of zones", key));
		if (chrono.t.size() > 1 && chrono.t.back() <= chrono.t[chrono.t.size() - 2])
			throw std::runtime_error(std::format("{}: chronology times must increase", key));
	}
	if (g.value("unit", std::string("m3/s")) == "mm/h")
	{
		std::vector<double> area(chrono.zoneCount(), 0.0);
		for (auto zk : chrono.zones)
			if (zk >= 1 && zk <= area.size())
				area[zk - 1] += c.dx() * c.dy();
		for (auto &line : chrono.q)
			for (size_t k = 0; k < line.size(); ++k)
				line[k] *= 1e-3 / 3600.0 * area[k];
	}
}

json chronologyJson(CaseData::Chronology const &chrono, char const *zonesKey)
{
	json lines = json::array();
	for (size_t l = 0; l < chrono.t.size(); ++l)
		lines.push_back({chrono.t[l], chrono.q[l]});
	return {{"zones", zonesKey}, {"lerp", chrono.lerp}, {"chronology", lines}};
}

json groundwaterJson(CaseData const &c)
{
	return {{"datum", c.gw.wolfgpu ? "wolfgpu" : "physical"},
	        {"kv", "gw_kv"},
	        {"kh", "gw_kh"},
	        {"sy", "gw_sy"},
	        {"z_sub", "gw_z_sub"},
	        {"h0", "gw_h0"}};
}

}        // namespace

// "zones": [{"kind": "source|sink|level", "value": 12.5, "label": "...", "rect": [i0, j0, i1, j1]}, ...]
// + npz "zones_mask" (uint8, zone index + 1 per cell)
void readZones(CaseData &c, json const &j, std::map<std::string, npy::Array> const &npz)
{
	c.zones.clear();
	c.zoneMask.clear();
	if (!j.contains("zones"))
		return;
	for (auto const &z : j["zones"])
	{
		CaseData::ZoneDef d;
		auto kind = z.value("kind", "source");
		d.kind    = kind == "sink" ? ZoneKind::Sink : kind == "level" ? ZoneKind::Level : ZoneKind::Source;
		d.value   = z.value("value", 0.0);
		d.label   = z.value("label", kind);
		if (z.contains("rect") && z["rect"].size() == 4)
		{
			d.i0 = z["rect"][0];
			d.j0 = z["rect"][1];
			d.i1 = z["rect"][2];
			d.j1 = z["rect"][3];
		}
		c.zones.push_back(d);
	}
	if (npz.count("zones_mask"))
	{
		auto m = npz.at("zones_mask").asFloat();
		if (m.size() == c.cells())
		{
			c.zoneMask.resize(m.size());
			for (size_t k = 0; k < m.size(); ++k)
				c.zoneMask[k] = static_cast<uint8_t>(m[k]);
		}
	}
	if (c.zoneMask.empty() && !c.zones.empty())
	{
		// rebuild from the rectangles
		c.zoneMask.assign(c.cells(), 0);
		for (size_t z = 0; z < c.zones.size(); ++z)
			for (uint32_t i = c.zones[z].i0; i <= std::min(c.zones[z].i1, c.nx - 1); ++i)
				for (uint32_t jj = c.zones[z].j0; jj <= std::min(c.zones[z].j1, c.ny - 1); ++jj)
					c.zoneMask[size_t(i) * c.ny + jj] = static_cast<uint8_t>(z + 1);
	}
}

json zonesJson(CaseData const &c)
{
	json arr = json::array();
	for (auto const &z : c.zones)
		arr.push_back({{"kind", z.kind == ZoneKind::Sink ? "sink" : z.kind == ZoneKind::Level ? "level" : "source"},
		               {"value", z.value},
		               {"label", z.label},
		               {"rect", {z.i0, z.j0, z.i1, z.j1}}});
	return arr;
}

namespace {

json boundariesJson(CaseData const &c)
{
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
	return bc;
}

// Arrays of a saved case (npz): bed, initial depth, optional outside / zone masks
std::map<std::string, npy::Array> caseArrays(CaseData const &c)
{
	std::map<std::string, npy::Array> arrays;
	arrays["zb"] = npy::makeArray(c.zb, {c.nx, c.ny});
	arrays["h0"] = npy::makeArray(c.h0, {c.nx, c.ny});
	if (!c.outside.empty())
		arrays["outside"] = npy::makeArray(c.outside, {c.nx, c.ny});
	if (!c.zoneMask.empty())
		arrays["zones_mask"] = npy::makeArray(c.zoneMask, {c.nx, c.ny});
	if (c.gw.enabled)
	{
		arrays["gw_kv"]    = npy::makeArray(c.gw.kv, {c.nx, c.ny});
		arrays["gw_kh"]    = npy::makeArray(c.gw.kh, {c.nx, c.ny});
		arrays["gw_sy"]    = npy::makeArray(c.gw.sy, {c.nx, c.ny});
		arrays["gw_z_sub"] = npy::makeArray(c.gw.z_sub, {c.nx, c.ny});
		arrays["gw_h0"]    = npy::makeArray(c.gw.h0, {c.nx, c.ny});
	}
	auto zonesF = [&](CaseData::Chronology const &ch) { return std::vector<float>(ch.zones.begin(), ch.zones.end()); };
	if (c.infiltration.active())
		arrays["infiltration_zones"] = npy::makeArray(zonesF(c.infiltration), {c.nx, c.ny});
	if (c.rain.active())
		arrays["rain_zones"] = npy::makeArray(zonesF(c.rain), {c.nx, c.ny});
	return arrays;
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
		auto crs    = j.value("crs", std::string("EPSG:31370"));
		if (crs.rfind("EPSG:", 0) == 0)
			g.epsg = std::stoi(crs.substr(5));
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

	npy::saveNpz(assets / "data" / (c.name + ".npz"), caseArrays(c));

	json j{{"label", c.label},
	       {"type", "imported"},
	       {"data", "data/" + c.name + ".npz"},
	       {"cell", meta.cell},
	       {"origin", {meta.x0, meta.y0}},
	       {"crs", "EPSG:" + std::to_string(meta.epsg)},
	       {"vertical_datum", "DNG (approx. for foreign sources)"},
	       {"z_ref", c.z_ref},
	       {"eta_0", c.eta_0},
	       {"manning_n", c.manning_n},
	       {"cfl", c.cfl},
	       {"scheme", c.scheme},
	       {"drain_passes", c.drain_passes},
	       {"bc", boundariesJson(c)},
	       {"render", {{"depth_ref", c.depth_ref}, {"z_scale_0", c.z_scale_0}}},
	       {"sources", meta.sources},
	       {"attributions", meta.attributions},
	       {"zones", zonesJson(c)}};
	if (!meta.orthoFile.empty())
		j["orthophoto"] = meta.orthoFile;
	if (c.gw.enabled)
		j["groundwater"] = groundwaterJson(c);
	if (c.infiltration.active())
		j["infiltration"] = chronologyJson(c.infiltration, "infiltration_zones");
	if (c.rain.active())
		j["rain"] = chronologyJson(c.rain, "rain_zones");
	std::ofstream(assets / "cases" / (c.name + ".json")) << j.dump(2) << '\n';
}

void updateImportedCase(CaseData const &c, std::string const &assetDir)
{
	fs::path assets(assetDir);
	auto     path = assets / "cases" / (c.name + ".json");
	json     j    = fs::exists(path) ? readJson(path) : json::object();
	if (j.value("type", "") != "imported")
		throw std::runtime_error("only imported cases can be saved in place");

	npy::saveNpz(assets / j.at("data").get<std::string>(), caseArrays(c));
	j["zones"]     = zonesJson(c);
	j["bc"]        = boundariesJson(c);
	j["manning_n"] = c.manning_n;
	j["cfl"]       = c.cfl;
	j["scheme"]    = c.scheme;
	j["drain_passes"] = c.drain_passes;
	j["eta_0"]     = c.eta_0;
	j["z_ref"]     = c.z_ref;
	if (c.gw.enabled)
		j["groundwater"] = groundwaterJson(c);
	else
		j.erase("groundwater");
	for (auto [key, chrono] : {std::pair{"infiltration", &c.infiltration}, std::pair{"rain", &c.rain}})
		if (chrono->active())
			j[key] = chronologyJson(*chrono, std::string(key) == "rain" ? "rain_zones" : "infiltration_zones");
		else
			j.erase(key);
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
	readGroundwater(c, j, assets);
	readChronology(c, c.infiltration, "infiltration", j, assets);
	readChronology(c, c.rain, "rain", j, assets);

	for (float z : c.zb)
		if (!std::isfinite(z))
			throw std::runtime_error("case '" + name + "': non-finite bed elevation");
	return c;
}

CaseData makeCase(std::string const &name, std::string const &assetDir, CaseOverrides const &overrides)
{
	CaseData c = makeCase(name, assetDir);
	if (!overrides.scheme.empty())
		c.scheme = overrides.scheme;
	if (overrides.drainPasses >= 0)
		c.drain_passes = std::min(overrides.drainPasses, MAX_DRAIN_PASSES);
	return c;
}

// --assets, else assets/ next to the executable (packaged application), else in the working directory, else
// the source tree (development build)
std::string resolveAssetDir(std::string const &requested)
{
	if (!requested.empty())
		return requested;
	if (auto next = paths::exeDir() / "assets"; fs::exists(next / "cases"))
		return next.string();
	if (fs::exists("assets"))
		return "assets";
	return SWE_ASSET_DIR;
}

}        // namespace swe
