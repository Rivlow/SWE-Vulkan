#pragma once
// Port of solver/runge_kutta.py: explicit Runge-Kutta schemes as Butcher tableaux.
//   Q_s     = Q^n + dt * sum_{j<s} A[s][j] * K_j      K_s = L(Q_s)
//   Q^{n+1} = Q^n + dt * sum_s    b[s]    * K_s
// (autonomous system: the c_s are not needed)
#include <map>
#include <string>
#include <vector>

namespace swe {

struct ButcherTableau
{
	std::string                      label;
	int                              order = 1;
	std::vector<std::vector<double>> A;        // strictly lower triangular rows, A[0] = {}
	std::vector<double>              b;

	int stages() const { return static_cast<int>(b.size()); }

	// Is K_s used at all? (e.g. the FSAL last stage of Dormand-Prince is not)
	bool is_used(int s) const
	{
		if (b[s] != 0.0)
			return true;
		for (auto const &row : A)
			if (static_cast<int>(row.size()) > s && row[s] != 0.0)
				return true;
		return false;
	}
};

// euler, ssprk2, ssprk3, rk4, dopri5 (same values as the Python version)
std::map<std::string, ButcherTableau> const &schemes();
ButcherTableau const                        &scheme(std::string const &name);

}        // namespace swe
