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
			case WaterEdit::Level: h = depthForLevel(value, case_.zb[k]); break;
			case WaterEdit::Add: h = std::max(h + value, 0.0f); break;
			case WaterEdit::Clear: h = 0.0f; break;
			}
			case_.h0[k]  = h;
			case_.hu0[k] = 0.0f;
			case_.hv0[k] = 0.0f;
		}
}

// ---- cell sets ----------------------------------------------------------------------------------------------

CellMask Session::maskOf(CellRect const &rect) const
{
	CellMask m(case_.cells(), 0);
	CellRect r = rect;
	r.i1       = std::min(r.i1, case_.nx - 1);
	r.j1       = std::min(r.j1, case_.ny - 1);
	if (!r.empty())
		for (uint32_t i = r.i0; i <= r.i1; ++i)
			std::fill_n(m.begin() + static_cast<std::ptrdiff_t>(case_.index(i, r.j0)), r.j1 - r.j0 + 1, uint8_t{1});
	return m;
}

CellRect Session::boundsOf(CellMask const &cells) const
{
	CellRect b{};        // empty
	if (!validMask(cells))
		return b;
	bool any = false;
	for (uint32_t i = 0; i < case_.nx; ++i)
		for (uint32_t j = 0; j < case_.ny; ++j)
			if (cells[case_.index(i, j)])
			{
				if (!any)
					b = {i, j, i, j};
				else
				{
					b.i0 = std::min(b.i0, i);
					b.j0 = std::min(b.j0, j);
					b.i1 = std::max(b.i1, i);
					b.j1 = std::max(b.j1, j);
				}
				any = true;
			}
	return b;
}

size_t Session::countOf(CellMask const &cells) const
{
	return validMask(cells) ? static_cast<size_t>(std::ranges::count_if(cells, [](uint8_t v) { return v != 0; })) : 0;
}

void Session::editWater(CellMask const &cells, WaterEdit mode, float value)
{
	if (!validMask(cells))
		return;
	bool any = false;
	for (size_t k = 0; k < cells.size(); ++k)
	{
		if (!cells[k])
			continue;
		any     = true;
		float h = case_.h0[k];
		switch (mode)
		{
		case WaterEdit::Depth: h = value; break;
		case WaterEdit::Level: h = depthForLevel(value, case_.zb[k]); break;
		case WaterEdit::Add: h = std::max(h + value, 0.0f); break;
		case WaterEdit::Clear: h = 0.0f; break;
		}
		case_.h0[k]  = h;
		case_.hu0[k] = 0.0f;
		case_.hv0[k] = 0.0f;
	}
	if (any)
		solver_.reset(case_);        // the live state is the initial state (the Conditions stage sits at t = 0)
}

bool Session::addZone(CellRect const &rect, ZoneKind kind, double value, std::string const &label)
{
	return addZone(maskOf(rect), kind, value, label);
}

bool Session::addZone(CellMask const &cells, ZoneKind kind, double value, std::string const &label)
{
	if (!validMask(cells))
		return false;
	CellRect r = boundsOf(cells);
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
	for (size_t k = 0; k < cells.size(); ++k)
		if (cells[k])
			case_.zoneMask[k] = id;
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

// ---- weak face conditions -----------------------------------------------------------------------------------

bool Session::addFaceBc(CellMask const &cells, Side side, BcKind kind, double value, std::string const &label, bool allFaces)
{
	if (!validMask(cells))
		return false;
	if (case_.faceBcs.size() >= MAX_FACE_BC)
	{
		log::warn("at most {} groups of face conditions", MAX_FACE_BC);
		return false;
	}
	CaseData::FaceBcDef def{.kind = kind, .value = value, .label = label};
	for (uint32_t i = 0; i < case_.nx; ++i)
		for (uint32_t j = 0; j < case_.ny; ++j)
		{
			uint32_t k = static_cast<uint32_t>(case_.index(i, j));
			if (!cells[k])
				continue;
			if (!allFaces)
			{
				// skip the faces shared with another selected cell: only the envelope of the set is a border
				bool shared = false;
				switch (side)
				{
				case SIDE_WEST: shared = i > 0 && cells[case_.index(i - 1, j)]; break;
				case SIDE_EAST: shared = i + 1 < case_.nx && cells[case_.index(i + 1, j)]; break;
				case SIDE_SOUTH: shared = j > 0 && cells[case_.index(i, j - 1)]; break;
				case SIDE_NORTH: shared = j + 1 < case_.ny && cells[case_.index(i, j + 1)]; break;
				}
				if (shared)
					continue;
			}
			def.faces.push_back(faceOf(k, side));
		}
	if (def.faces.empty())
		return false;
	// a face belongs to one group: remove the new faces from the older groups (empty groups disappear)
	std::vector<uint8_t> taken(case_.cells() * 4, 0);
	for (uint32_t f : def.faces)
		taken[f] = 1;
	for (auto &g : case_.faceBcs)
		std::erase_if(g.faces, [&](uint32_t f) { return taken[f] != 0; });
	std::erase_if(case_.faceBcs, [](CaseData::FaceBcDef const &g) { return g.faces.empty(); });
	case_.faceBcs.push_back(std::move(def));
	solver_.uploadFaceBcs(case_);
	applyParameters();        // FLAG_FACE_BC, step structure
	return true;
}

bool Session::addBridge(CellMask const &cells, double zRoof, double zDeck, std::string const &label)
{
	if (!validMask(cells) || countOf(cells) == 0)
		return false;
	CaseData::BridgeDef b{.label = label.empty() ? "pont" : label, .zRoof = zRoof, .zDeck = std::max(zDeck, zRoof), .cells = cells};
	// a cell is under one deck only: the new one takes its cells from the others
	for (auto &o : case_.bridges)
		for (size_t k = 0; k < cells.size(); ++k)
			if (cells[k])
				o.cells[k] = 0;
	std::erase_if(case_.bridges, [&](CaseData::BridgeDef const &o) { return countOf(o.cells) == 0; });
	case_.bridges.push_back(std::move(b));
	applyBridges();
	return true;
}

void Session::removeBridge(size_t index)
{
	if (index >= case_.bridges.size())
		return;
	case_.bridges.erase(case_.bridges.begin() + static_cast<std::ptrdiff_t>(index));
	applyBridges();
}

void Session::setBridge(size_t index, double zRoof, double zDeck)
{
	if (index >= case_.bridges.size())
		return;
	case_.bridges[index].zRoof = zRoof;
	case_.bridges[index].zDeck = std::max(zDeck, zRoof);
	applyBridges();
}

void Session::setBridgeSlot(double slot)
{
	case_.bridgeSlot = std::clamp(slot, 1e-3, 1.0);
	applyParameters();
}

void Session::applyBridges()
{
	case_.rebuildBridgeArrays();
	solver_.uploadBridges(case_);
	applyParameters();
	solver_.reset(case_);        // the stored depths change with the decks: back to t = 0
}

void Session::removeFaceBc(size_t index)
{
	if (index >= case_.faceBcs.size())
		return;
	case_.faceBcs.erase(case_.faceBcs.begin() + static_cast<std::ptrdiff_t>(index));
	solver_.uploadFaceBcs(case_);
	applyParameters();
}

void Session::setFaceBc(size_t index, BcKind kind, double value)
{
	if (index >= case_.faceBcs.size())
		return;
	case_.faceBcs[index].kind  = kind;
	case_.faceBcs[index].value = value;
	solver_.uploadFaceBcs(case_);
	applyParameters();
}

CellMask Session::faceBcCells(size_t index) const
{
	CellMask m(case_.cells(), 0);
	if (index < case_.faceBcs.size())
		for (uint32_t f : case_.faceBcs[index].faces)
			if (faceCell(f) < m.size())
				m[faceCell(f)] = 1;
	return m;
}

CaseData::Chronology &Session::chronology(ChronologyKind kind)
{
	return kind == ChronologyKind::Rain ? case_.rain : case_.infiltration;
}

bool Session::addChronologyZone(ChronologyKind kind, CellRect const &rect, double q)
{
	return addChronologyZone(kind, maskOf(rect), q);
}

bool Session::addChronologyZone(ChronologyKind kind, CellMask const &cells, double q)
{
	if (!validMask(cells) || countOf(cells) == 0)
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
	for (size_t k = 0; k < cells.size(); ++k)
		if (cells[k])
			ch.zones[k] = static_cast<uint16_t>(id);
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
		setSoil(CellRect{0, 0, case_.nx - 1, case_.ny - 1}, *params);
		return;        // setSoil reloads
	}
	solver_.load(case_);
	updateBedRange();
}

void Session::setSoil(CellRect const &rect, SoilParams const &p)
{
	setSoil(maskOf(rect), p);
}

void Session::setSoil(CellMask const &cells, SoilParams const &p)
{
	if (!case_.gw.enabled || !validMask(cells) || countOf(cells) == 0)
		return;
	auto &gw = case_.gw;
	for (size_t k = 0; k < cells.size(); ++k)
	{
		if (!cells[k])
			continue;
		float zb    = case_.zb[k];
		float thick = static_cast<float>(std::max(p.depth, 0.01));
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
			case_.h0[k] = depthForLevel(case_.zb[k] + case_.h0[k], zb[k]);
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
