#pragma once
// Native dialogs of the platform (Windows: COM IFileOpenDialog). Everything is a no-op elsewhere.
#include <filesystem>
#include <optional>
#include <string>

namespace swe::dialog {

// Modal folder picker, started in `initial` when it exists; nullopt when cancelled or unsupported
std::optional<std::filesystem::path> pickFolder(std::filesystem::path const &initial, std::wstring const &title);

// Opens the folder in the file explorer (created first when missing); false when unsupported or failed
bool openInExplorer(std::filesystem::path const &dir);

}        // namespace swe::dialog
