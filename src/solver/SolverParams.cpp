#include "solver/SolverParams.h"

#include "cases/Case.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace swe {

SolverParams buildSolverParams(CaseData const &c)
{
	// Constants folded in double like Taichi folds the Python expressions, then rounded to f32.
	double const dx  = c.dx(), dy = c.dy();
	double const K_v = 3.0;        // K_VENKATAKRISHNAN
	SolverParams p{};
	p.nx      = c.nx;
	p.ny      = c.ny;
	p.n_cells = static_cast<uint32_t>(c.cells());
	p.dx      = static_cast<float>(dx);
	p.dy      = static_cast<float>(dy);
	p.g       = static_cast<float>(c.g);
	p.cfl_dx  = static_cast<float>(c.cfl * std::min(dx, dy));
	p.h_eps   = static_cast<float>(c.h_eps);
	p.h_dry   = static_cast<float>(c.h_dry);
	p.h_dry4  = static_cast<float>(std::pow(c.h_dry, 4));
	p.eps2    = static_cast<float>(std::pow(K_v * std::min(dx, dy) * 10.0 / std::max(c.Lx, c.Ly), 3));
	p.g_n2    = static_cast<float>(c.g * c.manning_n * c.manning_n);
	// a (nearly) dry domain has no wave speed: cap dt as if 5 cm of water were present everywhere, so the
	// first drop of water is resolved and the clock stays meaningful (no effect on the reference cases)
	p.dt_max  = static_cast<float>(c.cfl * std::min(dx, dy) / std::sqrt(c.g * 0.05));
	p.flags   = (c.zones.empty() ? 0u : FLAG_ZONES) | (c.manning_n > 0.0 ? FLAG_FRICTION : 0u) |
	            (c.legacyInflow ? 0u : FLAG_EXACT_INFLOW) | (c.legacyLimiter ? FLAG_LEGACY_LIMITER : 0u) |
	            (c.reference ? FLAG_NO_CARRY : 0u);
	// lowest bed point of each side: a dry inflow side feeds the cells near it (river channel)
	auto sideMin = [&](int s) {
		float m = std::numeric_limits<float>::infinity();
		if (s < 2)
			for (uint32_t j = 0; j < c.ny; ++j)
				m = std::min(m, c.zb[c.index(s == 0 ? 0 : c.nx - 1, j)]);
		else
			for (uint32_t i = 0; i < c.nx; ++i)
				m = std::min(m, c.zb[c.index(i, s == 2 ? 0 : c.ny - 1)]);
		return std::isfinite(m) ? m : 0.0f;
	};
	if (c.gw.enabled)
	{
		p.flags |= FLAG_GROUNDWATER | (c.gw.wolfgpu ? FLAG_GW_WOLFGPU : 0u);
		if (std::ranges::any_of(c.gw.kh, [](float k) { return k > 0.0f; }))
			p.flags |= FLAG_GW_LATERAL;
		// explicit aquifer flow: dt <= Sy / (4 T (1/dx^2 + 1/dy^2)), the GPU divides by max(T / Sy)
		p.gw_dt_coef = static_cast<float>(0.25 / (1.0 / (dx * dx) + 1.0 / (dy * dy)));
	}
	if (c.infiltration.active() || c.rain.active())
		p.flags |= FLAG_SOURCES;
	if (!c.faceBcs.empty())
		p.flags |= FLAG_FACE_BC;
	if (c.hasBridges())
		p.flags |= FLAG_BRIDGE;
	p.bridge_slot = static_cast<float>(c.bridgeSlot);
	p.bc_fill = 0.5f;
	for (int s = 0; s < 4; ++s)
	{
		p.bc_kind[s]  = static_cast<uint32_t>(c.bc[s].kind);
		p.bc_value[s] = static_cast<float>(c.bc[s].value);
		p.bc_zmin[s]  = sideMin(s);
		if (c.bc[s].kind == BcKind::Inflow)
			p.flags |= FLAG_INFLOW;
	}
	return p;
}

}        // namespace swe
