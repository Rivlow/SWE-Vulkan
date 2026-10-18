#include "solver/RungeKutta.h"

#include <stdexcept>

namespace swe {

std::map<std::string, TimeScheme> const &schemes()
{
	// Same methods as the Python reference (Butcher form there); the Shu-Osher coefficients below are the
	// classic optimal SSP ones (Gottlieb & Shu 1998).
	static std::map<std::string, TimeScheme> const table{
	    {"euler", {"Euler explicite", 1, {{0.0, 1.0}}}},
	    {"ssprk2", {"SSP-RK2 (Heun)", 2, {{0.0, 1.0}, {0.5, 0.5}}}},
	    {"ssprk3", {"SSP-RK3 (Shu-Osher)", 3, {{0.0, 1.0}, {0.75, 0.25}, {1.0 / 3, 2.0 / 3}}}},
	};
	return table;
}

TimeScheme const &scheme(std::string const &name)
{
	auto it = schemes().find(name);
	if (it == schemes().end())
		throw std::runtime_error("unknown time scheme: " + name + " (euler, ssprk2, ssprk3)");
	return it->second;
}

}        // namespace swe
