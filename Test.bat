@echo off

REM Ensure obj/bin directories exist

if not exist obj mkdir obj
if not exist bin mkdir bin

REM Delete previous build files

del /q /s "obj\*"
del /q /s "bin\*"

REM Copy required runtime files

copy %TBB_REDIST%\tbb12.dll bin
xcopy /Y /E /I src\cpp\kernels bin\%OPENCL_KERNELS_PATH%

REM Build C++ Test Applications

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.Test.Sequential.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\**.cpp ^
    /I %TBB_INCLUDE% ^
    /I %OPENCL_INCLUDE% ^
    /O2 ^
    /Ot ^
    /Ob2 ^
    /fp:precise ^
    /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
    /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
    /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
    /D EXECUTION_POLICY=Sequential ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.Test.Parallel.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\**.cpp ^
    /I %TBB_INCLUDE% ^
    /I %OPENCL_INCLUDE% ^
    /O2 ^
    /Ob3 ^
    /Ot ^
    /fp:fast ^
    /GL ^
    /Gy ^
    /Gw ^
    /arch:AVX2 ^
    /DNDEBUG ^
    /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
    /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
    /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
    /D EXECUTION_POLICY=Parallel ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /LTCG ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.Test.Accelerated.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\**.cpp ^
    /I %TBB_INCLUDE% ^
    /I %OPENCL_INCLUDE% ^
    /O2 ^
    /Ob3 ^
    /Ot ^
    /fp:fast ^
    /GL ^
    /Gy ^
    /Gw ^
    /arch:AVX2 ^
    /DNDEBUG ^
    /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
    /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
    /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
    /D EXECUTION_POLICY=Accelerated ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /LTCG ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

REM Run Tests

echo.
echo ================================
echo Running Tests...
echo ================================
echo.

echo === Sequential Execution Policy ===
bin\KQS.Test.Sequential.exe
echo.
echo === Parallel Execution Policy ===
bin\KQS.Test.Parallel.exe
echo.
echo === Accelerated Execution Policy ===
bin\KQS.Test.Accelerated.exe
