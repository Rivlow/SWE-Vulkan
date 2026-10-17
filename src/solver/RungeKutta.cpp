#include "solver/RungeKutta.h"

#include <stdexcept>

namespace swe {

std::map<std::string, ButcherTableau> const &schemes()
{
	static std::map<std::string, ButcherTableau> const table{
	    {"euler", {"Euler explicite", 1, {{}}, {1.0}}},

	    // SSP (Strong Stability Preserving): convex combinations of Euler steps -> keep h >= 0
	    // under the same CFL condition as Euler. Preferred for flows with dry fronts.
	    {"ssprk2", {"SSP-RK2 (Heun)", 2, {{}, {1.0}}, {0.5, 0.5}}},

	    {"ssprk3", {"SSP-RK3 (Shu-Osher)", 3, {{}, {1.0}, {0.25, 0.25}}, {1.0 / 6, 1.0 / 6, 2.0 / 3}}},

	    // Not SSP: more accurate on smooth solutions, but no h >= 0 guarantee
	    {"rk4", {"RK4 classique", 4, {{}, {0.5}, {0.0, 0.5}, {0.0, 0.0, 1.0}}, {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6}}},

	    {"dopri5",
	     {"Dormand-Prince 5 (pas fixe)",
	      5,
	      {{},
	       {1.0 / 5},
	       {3.0 / 40, 9.0 / 40},
	       {44.0 / 45, -56.0 / 15, 32.0 / 9},
	       {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
	       {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
	       {35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84}},
	      {35.0 / 384, 0.0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0.0}}},
	};
	return table;
}

ButcherTableau const &scheme(std::string const &name)
{
	auto it = schemes().find(name);
	if (it == schemes().end())
		throw std::runtime_error("unknown time scheme: " + name);
	return it->second;
}

}        // namespace swe
