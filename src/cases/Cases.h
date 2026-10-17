#pragma once
#include "cases/Case.h"

#include <string>
#include <vector>

namespace swe {

// Builds a case by name ("dam_break", ...). Data files are looked up under assetDir/data.
CaseData makeCase(std::string const &name, std::string const &assetDir);

std::vector<std::string> caseNames();

// Individual builders
CaseData makeDamBreak();
CaseData makeLakeAtRest();

}        // namespace swe
