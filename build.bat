@echo off
rem Usage: build.bat [debug|release] [extra cmake --build args]
setlocal
cd /d "%~dp0"
set PRESET=%1
if "%PRESET%"=="" set PRESET=debug
rem UTF-8 console: cl prints its /showIncludes notes in UTF-8 ("Remarque : inclusion du fichier :" with U+00A0),
rem and CMake writes the prefix Ninja looks for in the console code page. With cp850 they never matched: no header
rem was tracked and editing a .h left stale objects (crashes).
chcp 65001 >nul
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake --preset %PRESET% || exit /b 1
cmake --build --preset %PRESET% %2 %3 %4 %5 || exit /b 1
endlocal
