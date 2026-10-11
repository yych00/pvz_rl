@echo off
setlocal
set "ROOT=%~dp0.."
set "ARCH=%~1"
if not defined ARCH set "ARCH=x64"
if /i not "%ARCH%"=="x64" if /i not "%ARCH%"=="Win32" exit /b 1
set "CMAKE=cmake"
where cmake >nul 2>nul
if errorlevel 1 (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_DIR=%%i"
)
if defined VS_DIR set "CMAKE=%VS_DIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
"%CMAKE%" -S "%ROOT%\native" -B "%ROOT%\build\native-%ARCH%" -G "Visual Studio 17 2022" -A %ARCH%
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%ROOT%\build\native-%ARCH%" --config Release --parallel
if errorlevel 1 exit /b 1
if /i "%ARCH%"=="x64" (
    if not exist "%ROOT%\bin" mkdir "%ROOT%\bin"
    copy /y "%ROOT%\build\native-x64\out\Release\pvz_gym_native.dll" "%ROOT%\bin\pvz_gym_native.dll" >nul
    if errorlevel 1 exit /b 1
)
echo Native build complete: %ARCH%
exit /b 0
