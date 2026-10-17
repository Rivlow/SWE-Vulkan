#include "cases/Cases.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace swe {

std::vector<std::string> caseNames()
{
	return {"dam_break", "lake_at_rest"};
}

CaseData makeCase(std::string const &name, std::string const &)
{
	if (name == "dam_break")
		return makeDamBreak();
	if (name == "lake_at_rest")
		return makeLakeAtRest();
	throw std::runtime_error("unknown case: " + name + " (meuse and vesdre arrive with milestone M3)");
}

// Port of cases/dam_break.py: free surface eta_0 behind the dam at x = Lx/2, dry bed downstream,
// with a bump in the bed. Computed in f32 with the same operation order as the Taichi kernel.
CaseData makeDamBreak()
{
	CaseData c;
	c.name      = "dam_break";
	c.nx        = 250;
	c.ny        = 250;
	c.Lx        = 10.0;
	c.Ly        = 10.0;
	c.eta_0     = 1.0;
	c.manning_n = 0.0;
	c.depth_ref = 1.0;
	c.z_scale_0 = 2.0;
	for (auto &b : c.bc)
		b = {BcKind::Wall, 0.0};

	auto const n = c.cells();
	c.zb.resize(n);
	c.h0.assign(n, 0.0f);
	c.hu0.assign(n, 0.0f);
	c.hv0.assign(n, 0.0f);

	float const dx   = static_cast<float>(c.dx());
	float const dy   = static_cast<float>(c.dy());
	float const half = static_cast<float>(c.Lx / 2);
	float const eta0 = static_cast<float>(c.eta_0);
	for (uint32_t i = 0; i < c.nx; ++i)
		for (uint32_t j = 0; j < c.ny; ++j)
		{
			float x  = (static_cast<float>(i) + 0.5f) * dx;        // cell centre
			float y  = (static_cast<float>(j) + 0.5f) * dy;
			float ex = x - 7.5f, ey = y - 5.0f;
			float z  = 0.6f * std::exp(-(ex * ex + ey * ey) / 0.8f);
			auto  k  = size_t(i) * c.ny + j;
			c.zb[k]  = z;
			if (x < half)
				c.h0[k] = std::max(eta0 - z, 0.0f);
		}
	return c;
}

// Well-balancedness test: dam_break bed, still water at eta = 0.5 everywhere. The bump (0.6 m) emerges as a
// dry island, so wet/dry fronts on a slope are exercised too. Water must stay exactly at rest.
CaseData makeLakeAtRest()
{
	CaseData c = makeDamBreak();
	c.name     = "lake_at_rest";
	c.eta_0    = 0.5;
	for (size_t k = 0; k < c.cells(); ++k)
		c.h0[k] = std::max(static_cast<float>(c.eta_0) - c.zb[k], 0.0f);
	return c;
}

}        // namespace swe
