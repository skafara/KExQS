@echo off
setlocal EnableDelayedExpansion

set RESULTS_DIR=results

REM Ensure directories exist

if not exist obj mkdir obj
if not exist bin mkdir bin
if not exist %RESULTS_DIR% mkdir %RESULTS_DIR%

REM Delete previous files

del /q /s "obj\*"
del /q /s "bin\*"
:: del /q /s "%RESULTS_DIR%\*"

REM Copy required runtime files

copy %TBB_REDIST%\tbb12.dll bin
xcopy /Y /E /I src\cpp\kernels bin\%OPENCL_KERNELS_PATH%

REM Build C++ Test Applications

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.TestTime.Sequential.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTime.cpp ^
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
    /D BENCHMARKING_ENABLED ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.TestTime.Parallel.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTime.cpp ^
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
    /D BENCHMARKING_ENABLED ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /LTCG ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.TestTime.Accelerated.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTime.cpp ^
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
    /D BENCHMARKING_ENABLED ^
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
bin\KQS.TestTime.Sequential.exe
echo.
echo === Parallel Execution Policy ===
bin\KQS.TestTime.Parallel.exe
echo.
echo === Accelerated Execution Policy ===
bin\KQS.TestTime.Accelerated.exe


cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.TestTimeWhole.Sequential.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTimeWhole.cpp ^
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
    /Febin\KQS.TestTimeWhole.Parallel.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTimeWhole.cpp ^
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
    /Febin\KQS.TestTimeWhole.Accelerated.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestTimeWhole.cpp ^
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

echo.
echo ================================
echo Running Tests...
echo ================================
echo.

echo === Sequential Execution Policy ===
bin\KQS.TestTimeWhole.Sequential.exe
echo.
echo === Parallel Execution Policy ===
bin\KQS.TestTimeWhole.Parallel.exe
echo.
echo === Accelerated Execution Policy ===
bin\KQS.TestTimeWhole.Accelerated.exe


set PYTHON=.venv\Scripts\python.exe
set SCRIPT_TEST_DISTRIBUTION=src\python\test\KQS.TestDistribution.py


cl  /std:c++20 /EHsc ^
    /Foobj\ ^
    /Febin\KQS.TestDistribution.exe ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    src\cpp\test\KQS.TestDistribution.cpp ^
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

bin\KQS.TestDistribution.exe
for %%F in (%RESULTS_DIR%\KQS.TestDistribution.*.RandomOrg.txt) do (
    set FILE_RANDOMORG=%%F
    set FILE_PHILOX=!FILE_RANDOMORG:.RandomOrg.txt=.Philox.txt!

    for %%A in ("%%~nF") do (
        set BASE=%%~A
    )

    set NAME=!BASE:KQS.TestDistribution.=!
    set NAME=!NAME:.RandomOrg=!

    echo.
    echo --------------------------------------------
    echo Testing !NAME!
    echo File RandomOrg !FILE_RANDOMORG!
    echo File Philox    !FILE_PHILOX!
    echo --------------------------------------------

    %PYTHON% %SCRIPT_TEST_DISTRIBUTION% !FILE_RANDOMORG! !FILE_PHILOX!

    echo.
)


set SCRIPT_TEST_TIME=src\python\test\KQS.TestTime.py

%PYTHON% %SCRIPT_TEST_TIME%
