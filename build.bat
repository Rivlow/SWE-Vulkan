@echo off
rem Usage: build.bat [debug|release] [extra cmake --build args]
setlocal
cd /d "%~dp0"
set PRESET=%1
if "%PRESET%"=="" set PRESET=debug
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cmake --preset %PRESET% || exit /b 1
cmake --build --preset %PRESET% %2 %3 %4 %5 || exit /b 1
endlocal
