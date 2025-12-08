@echo off

REM Libraries

REM Create ext directory

if not exist ext mkdir ext


REM oneTBB

set TBB_VERSION=2022.3.0
set TBB_URL=https://github.com/oneapi-src/oneTBB/releases/download/v%TBB_VERSION%/oneapi-tbb-%TBB_VERSION%-win.zip
set TBB_ZIP=ext\oneapi-tbb-%TBB_VERSION%-win.zip
set TBB_DIR=ext\oneTBB

if not exist %TBB_DIR% (
    powershell -Command "& {Invoke-WebRequest '%TBB_URL%' -OutFile '%TBB_ZIP%'}"
    powershell -Command "& {Expand-Archive '%TBB_ZIP%' 'ext'}"
    del %TBB_ZIP%
    for /d %%D in ("ext\oneapi-tbb-%TBB_VERSION%*") do (
        ren "%%D" oneTBB
    )
)

REM OpenCL

set OPENCL_VERSION=2025.07.23
set OPENCL_URL=https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v%OPENCL_VERSION%/OpenCL-SDK-v%OPENCL_VERSION%-Win-x64.zip
set OPENCL_ZIP=ext\OpenCL-SDK-v%OPENCL_VERSION%-Win-x64.zip
set OPENCL_DIR=ext\OpenCL

if not exist %OPENCL_DIR% (
    powershell -Command "& {Invoke-WebRequest '%OPENCL_URL%' -OutFile '%OPENCL_ZIP%'}"
    powershell -Command "& {Expand-Archive '%OPENCL_ZIP%' 'ext'}"
    del %OPENCL_ZIP%
    for /d %%D in ("ext\OpenCL-SDK-v%OPENCL_VERSION%*") do (
        ren "%%D" OpenCL
    )
)

REM Visual Studio Environment

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do (
    set VS_PATH=%%i
)

if not defined VS_PATH (
    echo Visual Studio not found.
    exit /b 1
)

REM Set up environment variables

call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Project Environment Variables

set TBB_INCLUDE=%TBB_DIR%\include
set TBB_LIB=%TBB_DIR%\lib\intel64\vc14
set TBB_REDIST=%TBB_DIR%\redist\intel64\vc14

set OPENCL_INCLUDE=%OPENCL_DIR%\include
set OPENCL_LIB=%OPENCL_DIR%\lib

set OPENCL_KERNELS_PATH=kernels

set RANDOMORG_FILES_PATH=data/randomorg

REM Done
cmd
