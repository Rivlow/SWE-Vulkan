@echo off
rem Builds the release, stages the packaged application (CMake target `dist`) and produces the installer
rem build\release\SWE-Vulkan-<version>-setup.exe with Inno Setup (winget install JRSoftware.InnoSetup).
rem   make_installer.bat         complete: with the imported demonstration area (zone_importee, ~27 MB)
rem   make_installer.bat light   without it: analytic cases only, < 25 MB (Discord), areas are imported online
setlocal
cd /d "%~dp0.."
call "%~dp0..\build.bat" release --target dist || exit /b 1

set SUFFIX=
if /i "%1"=="light" (
  del /q "build\release\dist\assets\data\zone_importee*" 2>nul
  del /q "build\release\dist\assets\cases\zone_importee.json" 2>nul
  set SUFFIX=-light
)

set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
  echo Inno Setup introuvable : winget install JRSoftware.InnoSetup
  exit /b 1
)
%ISCC% /Qp "/DSuffix=%SUFFIX%" "packaging\installer.iss" || exit /b 1
echo.
echo Installateur : build\release\SWE-Vulkan-*%SUFFIX%-setup.exe
endlocal
