#include "util/FileDialog.h"

#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>
#include <shellapi.h>
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")
#endif

#include <system_error>

namespace swe::dialog {

#ifdef _WIN32

namespace {

// COM apartment for the duration of a call (GLFW does not initialise COM for us)
struct ComScope
{
	HRESULT hr;
	ComScope() : hr(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)) {}
	~ComScope()
	{
		if (SUCCEEDED(hr))
			CoUninitialize();
	}
	bool ok() const { return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE; }
};

template <class T>
struct ComPtr
{
	T *p = nullptr;
	~ComPtr()
	{
		if (p)
			p->Release();
	}
	T  *operator->() const { return p; }
	T **out() { return &p; }
};

}        // namespace

std::optional<std::filesystem::path> pickFolder(std::filesystem::path const &initial, std::wstring const &title)
{
	ComScope com;
	if (!com.ok())
		return std::nullopt;
	ComPtr<IFileOpenDialog> dlg;
	if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(dlg.out()))))
		return std::nullopt;
	FILEOPENDIALOGOPTIONS options = 0;
	dlg->GetOptions(&options);
	dlg->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
	if (!title.empty())
		dlg->SetTitle(title.c_str());
	std::error_code ec;
	if (!initial.empty() && std::filesystem::is_directory(initial, ec))
	{
		ComPtr<IShellItem> folder;
		if (SUCCEEDED(SHCreateItemFromParsingName(initial.wstring().c_str(), nullptr, IID_PPV_ARGS(folder.out()))))
			dlg->SetFolder(folder.p);
	}
	if (FAILED(dlg->Show(GetActiveWindow())))
		return std::nullopt;        // cancelled
	ComPtr<IShellItem> item;
	if (FAILED(dlg->GetResult(item.out())))
		return std::nullopt;
	PWSTR name = nullptr;
	if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &name)) || !name)
		return std::nullopt;
	std::filesystem::path result(name);
	CoTaskMemFree(name);
	return result;
}

bool openInExplorer(std::filesystem::path const &dir)
{
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	if (!std::filesystem::is_directory(dir, ec))
		return false;
	auto r = ShellExecuteW(nullptr, L"explore", dir.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
	return reinterpret_cast<INT_PTR>(r) > 32;
}

#else

std::optional<std::filesystem::path> pickFolder(std::filesystem::path const &, std::wstring const &)
{
	return std::nullopt;
}

bool openInExplorer(std::filesystem::path const &)
{
	return false;
}

#endif

}        // namespace swe::dialog
