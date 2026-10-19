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

// Command-line choices taking precedence over the case file
struct CaseOverrides
{
	std::string scheme;               // empty = case value
	int         drainPasses = -1;     // < 0 = case value
	bool        legacyLimiter = false; // --legacy-limiter: Venkatakrishnan limiter of the Python reference
	bool        reference = false;     // --reference: everything as the Python reference (validation, see CaseData)
	double      hEps = -1.0, hDry = -1.0;        // --h-eps / --h-dry (m), < 0 = case value
};
CaseData makeCase(std::string const &name, std::string const &assetDir, CaseOverrides const &overrides);

// Asset directory: `requested` (--assets), else ./assets, else the source tree
std::string resolveAssetDir(std::string const &requested);

// Geographic origin of an imported case (Lambert 72), when known.
struct GeoOrigin
{
	bool        valid = false;
	double      x0 = 0, y0 = 0;        // lower-left corner (m)
	double      cell = 0;
	int         epsg = 31370;
	std::string orthoFile;             // relative to the asset dir, empty if none
};
GeoOrigin caseOrigin(std::string const &name, std::string const &assetDir);

// Saves a case built in the application (geographic import) as assets/cases/<name>.json +
// assets/data/<name>.npz (zb, h0, outside, bathy mask), with provenance metadata.
struct ImportMetadata
{
	double                   x0 = 0, y0 = 0, cell = 0;        // grid origin in `epsg`
	int                      epsg = 31370;
	std::vector<std::string> sources;                        // human-readable list
	std::vector<std::string> attributions;
	std::string              orthoFile;                      // png next to the npz, optional
};
void saveImportedCase(CaseData const &data, ImportMetadata const &meta, std::string const &assetDir);

// Rewrites the arrays and the physical parameters of an existing imported case (bed edits, boundary
// conditions...), keeping its provenance metadata.
void updateImportedCase(CaseData const &data, std::string const &assetDir);

}        // namespace swe
