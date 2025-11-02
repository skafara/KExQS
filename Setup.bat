@echo off

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do (
    set VS_PATH=%%i
)

if not defined VS_PATH (
    echo Visual Studio not found.
    exit /b 1
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
cmd
