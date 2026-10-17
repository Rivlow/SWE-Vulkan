#pragma once
#include "cases/Case.h"

#include <string>
#include <vector>

namespace swe {

struct CaseInfo
{
	std::string name;        // file stem of assets/cases/<name>.json
	std::string label;
};

// Cases described by assets/cases/*.json (sorted by name).
std::vector<CaseInfo> listCases(std::string const &assetDir);

// Builds a case from assets/cases/<name>.json. Data files are relative to assetDir.
CaseData makeCase(std::string const &name, std::string const &assetDir);

}        // namespace swe
