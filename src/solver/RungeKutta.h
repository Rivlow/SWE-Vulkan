#pragma once
// Explicit SSP Runge-Kutta schemes in Shu-Osher form: a sequence of Euler steps combined convexly with Q^n,
//   Q_0 = Q^n,   Q_{s+1} = a_s Q^n + b_s (Q_s + dt L(Q_s)),   Q^{n+1} = Q_S
// with a_s, b_s >= 0 and a_s + b_s = 1 (evaluated as Q^n + b_s (E(Q_s) - Q^n), see euler_stage in solver.slang).
// Whatever an Euler step preserves (h >= 0 with the drain limiter) the full step preserves too, exactly, in
// floating point as well. Only SSP schemes qualify, hence euler / ssprk2 / ssprk3 (rk4 and Dormand-Prince have
// no such form and were dropped).
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace swe {

struct TimeScheme
{
	std::string                            label;
	int                                    order = 1;
	std::vector<std::pair<double, double>> stages;        // (a_s, b_s)

	int nstages() const { return static_cast<int>(stages.size()); }
};

// euler, ssprk2, ssprk3
std::map<std::string, TimeScheme> const &schemes();
TimeScheme const                        &scheme(std::string const &name);

}        // namespace swe
