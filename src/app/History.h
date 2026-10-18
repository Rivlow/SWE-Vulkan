#pragma once
#include "solver/SolverParams.h"

#include <array>
#include <vector>

namespace swe {

// Time series of the plots (t on the x axis), halved when they grow too long
struct History
{
	std::vector<double>                         t, volume, wetArea, dt;
	std::vector<double>                         gwVolume, gwInfil, gwExfil;        // groundwater cases (m3)
	std::vector<double>                         rain;                              // rain volume since t = 0 (m3)
	std::array<std::vector<double>, MAX_PROBES> eta, depth, speed, gwHead;

	void clear();
	// appends the scalars of one frame; probes beyond `probes` are NaN (gaps)
	void push(SolverScalars const &s, double cellArea, size_t probes);
	// the probe set changed: previous probe series become gaps
	void clearProbes();

  private:
	void decimate();
};

}        // namespace swe
