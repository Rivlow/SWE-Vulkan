#include "geo/TerrainBuilder.h"

#include "geo/OnlineSources.h"
#include "util/Log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>

namespace swe::geo {

TerrainBuilder::TerrainBuilder(LocalLibrary &library) :
    library_(library)
{
	terrain_.push_back(std::make_unique<LocalSource>(library_, SourceKind::Terrain));
	for (auto &s : makeOnlineSources())
		terrain_.push_back(std::move(s));
	bathy_ = std::make_unique<LocalSource>(library_, SourceKind::Bathymetry);
}

void TerrainBuilder::refreshLocal()
{
	for (auto &s : terrain_)
		if (auto *local = dynamic_cast<LocalSource *>(s.get()))
			local->refresh();
	if (auto *local = dynamic_cast<LocalSource *>(bathy_.get()))
		local->refresh();
}

std::vector<SourceReport> TerrainBuilder::probe(BBox const &box, int epsg, Progress &progress, bool fetchCoarse)
{
	std::vector<SourceReport> reports;
	std::vector<RasterSource *> all;
	for (auto &s : terrain_)
		all.push_back(s.get());
	all.push_back(bathy_.get());

	double coarse = std::max(box.width(), box.height()) / 96.0;
	for (size_t k = 0; k < all.size(); ++k)
	{
		auto        *s = all[k];
		SourceReport r{.id = s->info().id, .name = s->info().name, .footprint = s->footprintCoverage(box, epsg)};
		// light probe: only local rasters are read (cheap); online services answer at the import
		if (fetchCoarse && r.footprint > 0.0 && !s->info().online)
		{
			progress.set(float(k) / float(all.size()), "Sondage : " + s->info().name);
			std::string err;
			Grid        g = s->fetch(box, std::max(coarse, s->info().resolution), epsg, progress, &err);
			r.data        = g.v.empty() ? 0.0 : g.coverage();
			r.error       = err;
		}
		reports.push_back(std::move(r));
		if (progress.cancelled())
			break;
	}
	return reports;
}

ImportResult TerrainBuilder::build(ImportRequest const &req, Progress &progress, std::string *error)
{
	ImportResult res;
	Grid         bed(req.box, req.cell, req.epsg);
	size_t const n = bed.cells();
	if (n == 0 || n > 16'000'000)
	{
		if (error)
			*error = std::format("grille de {} mailles : réduire la zone ou agrandir la maille", n);
		return res;
	}

	// terrain: fill by priority until every cell has data
	size_t step = 0, steps = terrain_.size() + 2;
	for (auto &s : terrain_)
	{
		if (progress.cancelled())
			return {};
		SourceReport r{.id = s->info().id, .name = s->info().name, .footprint = s->footprintCoverage(req.box, req.epsg)};
		double       missing = 1.0 - bed.coverage();
		if (r.footprint > 0.0 && missing > 1e-6)
		{
			progress.set(float(step) / float(steps), s->info().name);
			std::string err;
			auto        t0 = std::chrono::steady_clock::now();
			Grid        g  = s->fetch(req.box, req.cell, req.epsg, progress, &err);
			r.error        = err;
			log::info("  {:<40} {:.1f} s{}", s->info().name, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
			          err.empty() ? "" : " : " + err);
			if (!g.v.empty())
			{
				r.data        = g.coverage();
				size_t before = bed.countNoData();
				bed.fillFrom(g);
				r.used = bed.countNoData() < before;
				if (r.used)
					res.attributions.push_back(s->info().attribution);
			}
			else
				r.data = 0.0;
		}
		res.sources.push_back(std::move(r));
		++step;
	}
	if (bed.coverage() <= 0.0)
	{
		if (error)
			*error = "aucune source ne couvre cette zone";
		return {};
	}

	// bathymetry overrides the terrain where surveyed
	res.bathyMask.assign(n, 0);
	if (req.mergeBathy && bathy_->footprintCoverage(req.box, req.epsg) > 0.0)
	{
		progress.set(float(step) / float(steps), bathy_->info().name);
		std::string  err;
		Grid         g = bathy_->fetch(req.box, req.cell, req.epsg, progress, &err);
		SourceReport r{.id = bathy_->info().id, .name = bathy_->info().name, .footprint = bathy_->footprintCoverage(req.box, req.epsg), .error = err};
		if (!g.v.empty())
		{
			size_t count = 0;
			for (size_t k = 0; k < n; ++k)
				if (!g.isNoData(g.v[k]))
				{
					bed.v[k]         = g.v[k];
					res.bathyMask[k] = 1;
					++count;
				}
			r.data            = g.coverage();
			r.used            = count > 0;
			res.bathyCoverage = double(count) / double(n);
			if (r.used)
				res.attributions.push_back(bathy_->info().attribution);
		}
		res.sources.push_back(std::move(r));
	}
	++step;

	// holes: dilate the nearest data (walls like fill_outside) and remember where
	progress.set(float(step) / float(steps), "Bouchage des trous");
	if (req.fillHoles)
		res.outside = bed.fillHoles();
	else
	{
		res.outside.assign(n, 0);
		for (size_t k = 0; k < n; ++k)
			if (bed.isNoData(bed.v[k]))
			{
				res.outside[k] = 1;
				bed.v[k]       = 0.0f;
			}
	}
	std::sort(res.attributions.begin(), res.attributions.end());
	res.attributions.erase(std::unique(res.attributions.begin(), res.attributions.end()), res.attributions.end());
	res.bed = std::move(bed);
	progress.set(1.0f, "Terminé");
	return res;
}

CaseData TerrainBuilder::toCase(ImportResult const &result, ImportRequest const &req)
{
	CaseData c;
	auto const &bed = result.bed;
	c.name      = req.name;
	c.label     = req.label.empty() ? req.name : req.label;
	c.nx        = bed.nx;
	c.ny        = bed.ny;
	c.Lx        = bed.nx * bed.cell;
	c.Ly        = bed.ny * bed.cell;
	c.epsg      = bed.epsg;
	c.manning_n = req.manning;
	c.depth_ref = req.depthRef;
	c.z_scale_0 = req.zScale;
	c.scheme    = "ssprk2";
	for (auto &b : c.bc)
		b = {BcKind::Wall, 0.0};

	// relative elevations for f32 precision, like the Python cases
	float zmin = *std::min_element(bed.v.begin(), bed.v.end());
	c.z_ref    = std::floor(double(zmin));
	float zRef = static_cast<float>(c.z_ref);
	size_t n   = bed.cells();
	c.zb.resize(n);
	c.h0.assign(n, 0.0f);
	c.hu0.assign(n, 0.0f);
	c.hv0.assign(n, 0.0f);
	for (size_t k = 0; k < n; ++k)
		c.zb[k] = bed.v[k] - zRef;
	c.outside = result.outside;
	if (std::ranges::none_of(c.outside, [](uint8_t o) { return o; }))
		c.outside.clear();

	float level = static_cast<float>(req.level - c.z_ref);
	c.eta_0     = req.initial == InitialWater::Dry ? zmin - c.z_ref + req.depthRef : double(level);
	if (req.initial != InitialWater::Dry)
		for (uint32_t i = 0; i < c.nx; ++i)
			for (uint32_t j = 0; j < c.ny; ++j)
			{
				size_t k = size_t(i) * c.ny + j;
				if (req.initial == InitialWater::LevelInRectangle)
				{
					double x = bed.x0 + (i + 0.5) * bed.cell, y = bed.y0 + (j + 0.5) * bed.cell;
					if (!req.levelBox.contains(x, y))
						continue;
				}
				if (!c.outside.empty() && c.outside[k])
					continue;
				c.h0[k] = std::max(level - c.zb[k], 0.0f);
			}
	return c;
}

}        // namespace swe::geo
