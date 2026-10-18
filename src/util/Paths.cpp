#include "util/Paths.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace swe::paths {

std::filesystem::path exeDir()
{
#ifdef _WIN32
	wchar_t exe[MAX_PATH];
	if (GetModuleFileNameW(nullptr, exe, MAX_PATH) > 0)
		return std::filesystem::path(exe).parent_path();
#endif
	return std::filesystem::current_path();
}

std::filesystem::path dataDir()
{
	return exeDir();
}

}        // namespace swe::paths
