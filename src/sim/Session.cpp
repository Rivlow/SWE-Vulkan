#include "sim/Session.h"

#include "solver/RungeKutta.h"
#include "util/Log.h"

#include <algorithm>
#include <stdexcept>

namespace swe {

Session::Session(vkc::Context const &ctx) :
    solver_(ctx)
{
}

void Session::load(CaseData data)
{
	data.drain_passes = std::clamp(data.drain_passes, 0, MAX_DRAIN_PASSES);
	if (data.hu0.size() != data.cells())
		data.hu0.assign(data.cells(), 0.0f);
	if (data.hv0.size() != data.cells())
		data.hv0.assign(data.cells(), 0.0f);
	scheme(data.scheme);        // validates the name before anything is touched
	solver_.load(data);
	case_ = std::move(data);
	updateBedRange();
}

void Session::reset()
{
	solver_.reset(case_);
}

void Session::applyParameters()
{
	solver_.configure(case_);
}

void Session::setScheme(std::string const &name)
{
	if (name == case_.scheme)
		return;
	scheme(name);
	case_.scheme = name;
	applyParameters();
}

void Session::setDrainPasses(int passes)
{
	passes = std::clamp(passes, 0, MAX_DRAIN_PASSES);
	if (passes == case_.drain_passes)
		return;
	case_.drain_passes = passes;
	applyParameters();
}

void Session::setCfl(double cfl)
{
	case_.cfl = cfl;
	applyParameters();
}

void Session::setManning(double n)
{
	case_.manning_n = n;
	applyParameters();
}

void Session::setBoundary(int side, BoundaryCondition const &bc)
{
	case_.bc.at(side) = bc;
	applyParameters();
}

void Session::editWater(CellRect const &rect, WaterEdit mode, float value, bool alsoInitial)
{
	CellRect r = rect;
	r.i1 = std::min(r.i1, case_.nx - 1);
	r.j1 = std::min(r.j1, case_.ny - 1);
	if (r.empty())
		return;
	solver_.editWater(r, mode, value);
	if (!alsoInitial)
		return;
	// same rule as set_water in solver.slang, on the initial state
	for (uint32_t i = r.i0; i <= r.i1; ++i)
		for (uint32_t j = r.j0; j <= r.j1; ++j)
		{
			size_t k = case_.index(i, j);
			float  h = case_.h0[k];
			switch (mode)
			{
			case WaterEdit::Depth: h = value; break;
			case WaterEdit::Level: h = std::max(value - case_.zb[k], 0.0f); break;
			case WaterEdit::Add: h = std::max(h + value, 0.0f); break;
			case WaterEdit::Clear: h = 0.0f; break;
			}
			case_.h0[k]  = h;
			case_.hu0[k] = 0.0f;
			case_.hv0[k] = 0.0f;
		}
}

bool Session::addZone(CellRect const &rect, ZoneKind kind, double value, std::string const &label)
{
	CellRect r = rect;
	r.i1 = std::min(r.i1, case_.nx - 1);
	r.j1 = std::min(r.j1, case_.ny - 1);
	if (r.empty())
		return false;
	if (case_.zones.size() >= MAX_ZONES)
	{
		log::warn("at most {} zones", MAX_ZONES);
		return false;
	}
	if (case_.zoneMask.size() != case_.cells())
		case_.zoneMask.assign(case_.cells(), 0);
	case_.zones.push_back({.kind = kind, .value = value, .label = label, .i0 = r.i0, .j0 = r.j0, .i1 = r.i1, .j1 = r.j1});
	auto id = static_cast<uint8_t>(case_.zones.size());
	for (uint32_t i = r.i0; i <= r.i1; ++i)
		for (uint32_t j = r.j0; j <= r.j1; ++j)
			case_.zoneMask[case_.index(i, j)] = id;
	solver_.uploadZones(case_);
	applyParameters();        // FLAG_ZONES
	return true;
}

void Session::removeZone(size_t index)
{
	if (index >= case_.zones.size())
		return;
	case_.zones.erase(case_.zones.begin() + static_cast<std::ptrdiff_t>(index));
	// later zones move down by one
	for (auto &m : case_.zoneMask)
	{
		if (m == index + 1)
			m = 0;
		else if (m > index + 1)
			--m;
	}
	solver_.uploadZones(case_);
	applyParameters();
}

void Session::setZoneValue(size_t index, double value)
{
	if (index >= case_.zones.size())
		return;
	case_.zones[index].value = value;
	solver_.uploadZones(case_);
}

CaseData::Chronology &Session::chronology(ChronologyKind kind)
{
	return kind == ChronologyKind::Rain ? case_.rain : case_.infiltration;
}

bool Session::addChronologyZone(ChronologyKind kind, CellRect const &rect, double q)
{
	CellRect r = rect;
	r.i1 = std::min(r.i1, case_.nx - 1);
	r.j1 = std::min(r.j1, case_.ny - 1);
	if (r.empty())
		return false;
	auto &ch = chronology(kind);
	if (ch.zones.size() != case_.cells())
		ch.zones.assign(case_.cells(), 0);
	if (ch.t.empty())
	{
		ch.t = {0.0};
		ch.q = {{}};
	}
	// zones without any cell left (overwritten by later rectangles) keep their column: indices stay stable
	size_t id = ch.zoneCount() + 1;
	if (id > 0xffff)
		return false;
	for (auto &line : ch.q)
		line.push_back(q);
	for (uint32_t i = r.i0; i <= r.i1; ++i)
		for (uint32_t j = r.j0; j <= r.j1; ++j)
			ch.zones[case_.index(i, j)] = static_cast<uint16_t>(id);
	applyChronologies();
	return true;
}

void Session::removeChronologyZone(ChronologyKind kind, size_t zone)
{
	auto &ch = chronology(kind);
	if (zone >= ch.zoneCount())
		return;
	for (auto &line : ch.q)
		line.erase(line.begin() + static_cast<std::ptrdiff_t>(zone));
	for (auto &z : ch.zones)
	{
		if (z == zone + 1)
			z = 0;
		else if (z > zone + 1)
			--z;
	}
	if (ch.zoneCount() == 0)
		ch = {};
	applyChronologies();
}

void Session::setChronologyTable(ChronologyKind kind, std::vector<double> t, std::vector<std::vector<double>> q, bool lerp)
{
	auto &ch = chronology(kind);
	if (t.empty() || t.size() != q.size())
		return;
	for (size_t l = 0; l < t.size(); ++l)
		if (q[l].size() != ch.zoneCount() || (l > 0 && t[l] <= t[l - 1]))
			return;        // malformed (the GUI keeps times increasing)
	ch.t    = std::move(t);
	ch.q    = std::move(q);
	ch.lerp = lerp;
	applyChronologies();
}

std::vector<double> Session::chronologyZoneAreas(ChronologyKind kind) const
{
	auto const         &ch = kind == ChronologyKind::Rain ? case_.rain : case_.infiltration;
	std::vector<double> area(ch.zoneCount(), 0.0);
	for (auto z : ch.zones)
		if (z >= 1 && z <= area.size())
			area[z - 1] += case_.dx() * case_.dy();
	return area;
}

void Session::applyChronologies()
{
	solver_.setChronologies(case_);
	applyParameters();        // FLAG_SOURCES
}

void Session::setGroundwater(std::optional<SoilParams> const &params)
{
	auto &gw = case_.gw;
	if (!params)
	{
		gw = {};
	}
	else
	{
		size_t const n = case_.cells();
		gw.enabled     = true;
		gw.wolfgpu     = false;
		gw.kv.assign(n, 0.0f);
		gw.kh.assign(n, 0.0f);
		gw.sy.assign(n, 1.0f);
		gw.z_sub.assign(n, 0.0f);
		gw.h0.assign(n, 0.0f);
		soilDefaults_ = *params;
		setSoil({0, 0, case_.nx - 1, case_.ny - 1}, *params);
		return;        // setSoil reloads
	}
	solver_.load(case_);
	updateBedRange();
}

void Session::setSoil(CellRect const &rect, SoilParams const &p)
{
	if (!case_.gw.enabled)
		return;
	CellRect r = rect;
	r.i1 = std::min(r.i1, case_.nx - 1);
	r.j1 = std::min(r.j1, case_.ny - 1);
	if (r.empty())
		return;
	auto &gw = case_.gw;
	for (uint32_t i = r.i0; i <= r.i1; ++i)
		for (uint32_t j = r.j0; j <= r.j1; ++j)
		{
			size_t k    = case_.index(i, j);
			float  zb   = case_.zb[k];
			float  thick = static_cast<float>(std::max(p.depth, 0.01));
			gw.kv[k]    = static_cast<float>(std::max(p.kv, 0.0));
			gw.kh[k]    = static_cast<float>(std::max(p.kh, 0.0));
			gw.sy[k]    = static_cast<float>(std::clamp(p.sy, 1e-3, 1.0));
			gw.z_sub[k] = zb - thick;
			gw.h0[k]    = std::clamp(thick - static_cast<float>(p.tableDepth), 0.0f, thick);
		}
	solver_.load(case_);
	updateBedRange();
}

void Session::replaceBed(std::vector<float> zb)
{
	if (zb.size() != case_.cells())
		throw std::runtime_error("Session::replaceBed: grid mismatch");
	// still water keeps its surface: the initial depth follows the new bed
	for (size_t k = 0; k < case_.cells(); ++k)
		if (case_.h0[k] > 0.0f)
			case_.h0[k] = std::max(case_.zb[k] + case_.h0[k] - zb[k], 0.0f);
	case_.zb = std::move(zb);
	// the aquifer stays under the ground: wolfgpu's lies on the bed, a physical one is cut by a lowered bed
	auto &gw = case_.gw;
	if (gw.enabled)
		for (size_t k = 0; k < case_.cells(); ++k)
		{
			gw.z_sub[k] = gw.wolfgpu ? case_.zb[k] : std::min(gw.z_sub[k], case_.zb[k]);
			if (!gw.wolfgpu)
				gw.h0[k] = std::min(gw.h0[k], case_.zb[k] - gw.z_sub[k]);
		}
	solver_.load(case_);
	updateBedRange();
}

void Session::updateBedRange()
{
	if (case_.zb.empty())
		return;
	auto [mn, mx] = std::minmax_element(case_.zb.begin(), case_.zb.end());
	bedRange_     = {*mn, *mx};
	gwStats_      = {};
	auto const &gw = case_.gw;
	if (gw.enabled)
	{
		gwStats_.zSubMin      = *std::ranges::min_element(gw.z_sub);
		gwStats_.kvMax        = *std::ranges::max_element(gw.kv);
		gwStats_.thicknessMax = 0.0f;
		for (size_t k = 0; k < case_.cells(); ++k)
			gwStats_.thicknessMax = std::max(gwStats_.thicknessMax, case_.zb[k] - gw.z_sub[k]);
		gwStats_.thicknessMax = std::max(gwStats_.thicknessMax, 1e-3f);
	}
}

}        // namespace swe
