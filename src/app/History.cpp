#include "app/History.h"

#include <algorithm>
#include <cmath>

namespace swe {

namespace {

constexpr size_t kMaxSamples = 40000;

}        // namespace

void History::clear()
{
	for (auto *v : {&t, &volume, &wetArea, &dt, &gwVolume, &gwInfil, &gwExfil, &rain})
		v->clear();
	for (auto *series : {&eta, &depth, &speed, &gwHead})
		for (auto &v : *series)
			v.clear();
}

void History::clearProbes()
{
	for (auto *series : {&eta, &depth, &speed, &gwHead})
		for (auto &v : *series)
			std::fill(v.begin(), v.end(), NAN);
}

void History::push(SolverScalars const &s, double cellArea, size_t probes)
{
	t.push_back(s.t);
	volume.push_back(s.volume);
	wetArea.push_back(s.wetCells * cellArea);
	dt.push_back(s.dt);
	gwVolume.push_back(s.gwVolume);
	gwInfil.push_back(s.gwInfil);
	gwExfil.push_back(s.gwExfil);
	rain.push_back(s.srcRain);
	for (size_t p = 0; p < MAX_PROBES; ++p)
	{
		bool on = p < probes;
		eta[p].push_back(on ? s.probes[p].eta : NAN);
		depth[p].push_back(on ? s.probes[p].h : NAN);
		speed[p].push_back(on ? std::hypot(s.probes[p].u, s.probes[p].v) : NAN);
		gwHead[p].push_back(on ? s.probes[p].gwHead : NAN);
	}
	if (t.size() > kMaxSamples)
		decimate();
}

void History::decimate()
{
	auto half = [](std::vector<double> &v) {
		size_t k = 0;
		for (size_t i = 0; i < v.size(); i += 2)
			v[k++] = v[i];
		v.resize(k);
	};
	for (auto *v : {&t, &volume, &wetArea, &dt, &gwVolume, &gwInfil, &gwExfil, &rain})
		half(*v);
	for (auto *series : {&eta, &depth, &speed, &gwHead})
		for (auto &v : *series)
			half(v);
}

}        // namespace swe
