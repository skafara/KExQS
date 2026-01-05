@echo off
setlocal EnableDelayedExpansion

set RESULTS_DIR=results
set PYTHON=.venv\Scripts\python.exe

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

@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTime.Sequential.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTime.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ot ^
@REM     /Ob2 ^
@REM     /fp:precise ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Sequential ^
@REM     /D BENCHMARKING_ENABLED ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTime.Parallel.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTime.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ob3 ^
@REM     /Ot ^
@REM     /fp:fast ^
@REM     /GL ^
@REM     /Gy ^
@REM     /Gw ^
@REM     /arch:AVX2 ^
@REM     /DNDEBUG ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Parallel ^
@REM     /D BENCHMARKING_ENABLED ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /LTCG ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTime.Accelerated.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTime.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ob3 ^
@REM     /Ot ^
@REM     /fp:fast ^
@REM     /GL ^
@REM     /Gy ^
@REM     /Gw ^
@REM     /arch:AVX2 ^
@REM     /DNDEBUG ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Accelerated ^
@REM     /D BENCHMARKING_ENABLED ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /LTCG ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM REM Run Tests

@REM echo.
@REM echo ================================
@REM echo Running Tests...
@REM echo ================================
@REM echo.

@REM echo === Sequential Execution Policy ===
@REM bin\KQS.TestTime.Sequential.exe
@REM echo.
@REM echo === Parallel Execution Policy ===
@REM bin\KQS.TestTime.Parallel.exe
@REM echo.
@REM echo === Accelerated Execution Policy ===
@REM bin\KQS.TestTime.Accelerated.exe


@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTimeWhole.Sequential.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTimeWhole.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ot ^
@REM     /Ob2 ^
@REM     /fp:precise ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Sequential ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTimeWhole.Parallel.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTimeWhole.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ob3 ^
@REM     /Ot ^
@REM     /fp:fast ^
@REM     /GL ^
@REM     /Gy ^
@REM     /Gw ^
@REM     /arch:AVX2 ^
@REM     /DNDEBUG ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Parallel ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /LTCG ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestTimeWhole.Accelerated.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestTimeWhole.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ob3 ^
@REM     /Ot ^
@REM     /fp:fast ^
@REM     /GL ^
@REM     /Gy ^
@REM     /Gw ^
@REM     /arch:AVX2 ^
@REM     /DNDEBUG ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Accelerated ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /LTCG ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib

@REM echo.
@REM echo ================================
@REM echo Running Tests...
@REM echo ================================
@REM echo.

@REM echo === Sequential Execution Policy ===
@REM bin\KQS.TestTimeWhole.Sequential.exe
@REM echo.
@REM echo === Parallel Execution Policy ===
@REM bin\KQS.TestTimeWhole.Parallel.exe
@REM echo.
@REM echo === Accelerated Execution Policy ===
@REM bin\KQS.TestTimeWhole.Accelerated.exe


@REM set SCRIPT_TEST_DISTRIBUTION=src\python\test\KQS.TestDistribution.py


@REM cl  /std:c++20 /EHsc ^
@REM     /Foobj\ ^
@REM     /Febin\KQS.TestDistribution.exe ^
@REM     /I src\cpp\include ^
@REM     src\cpp\lib\**.cpp ^
@REM     src\cpp\test\KQS.TestDistribution.cpp ^
@REM     /I %TBB_INCLUDE% ^
@REM     /I %OPENCL_INCLUDE% ^
@REM     /O2 ^
@REM     /Ob3 ^
@REM     /Ot ^
@REM     /fp:fast ^
@REM     /GL ^
@REM     /Gy ^
@REM     /Gw ^
@REM     /arch:AVX2 ^
@REM     /DNDEBUG ^
@REM     /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
@REM     /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
@REM     /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
@REM     /D EXECUTION_POLICY=Accelerated ^
@REM     /link ^
@REM     /LIBPATH:%TBB_LIB% ^
@REM     /LIBPATH:%OPENCL_LIB% ^
@REM     /LTCG ^
@REM     /OPT:REF ^
@REM     tbb12.lib ^
@REM     OpenCL.lib


@REM set SCRIPT_TEST_TIME=src\python\test\KQS.TestTime.py
@REM %PYTHON% %SCRIPT_TEST_TIME%


@REM bin\KQS.TestDistribution.exe
@REM for %%F in (%RESULTS_DIR%\KQS.TestDistribution.*.RandomOrg.txt) do (
@REM     set FILE_RANDOMORG=%%F
@REM     set FILE_PHILOX=!FILE_RANDOMORG:.RandomOrg.txt=.Philox.txt!

@REM     for %%A in ("%%~nF") do (
@REM         set BASE=%%~A
@REM     )

@REM     set NAME=!BASE:KQS.TestDistribution.=!
@REM     set NAME=!NAME:.RandomOrg=!

@REM     echo.
@REM     echo --------------------------------------------
@REM     echo Testing !NAME!
@REM     echo File RandomOrg !FILE_RANDOMORG!
@REM     echo File Philox    !FILE_PHILOX!
@REM     echo --------------------------------------------

@REM     %PYTHON% %SCRIPT_TEST_DISTRIBUTION% !FILE_RANDOMORG! !FILE_PHILOX!

@REM     echo.
@REM )


set SCRIPT_TEST_DISTRIBUTION=src\python\test\KQS.TestDistribution.py

REM Loop over all Philox distribution files
for %%F in ("%RESULTS_DIR%\KQS.TestDistribution.*.Philox.txt") do (
    set "FILE_PHILOX=%%F"
    set "FILE_RANDOMORG=%%F"

    REM Replace suffix
    set "FILE_RANDOMORG=!FILE_RANDOMORG:.Philox.txt=.RandomOrg.txt!"

    echo ========================================
    echo Comparing:
    echo   !FILE_PHILOX!
    echo   !FILE_RANDOMORG!
    echo ----------------------------------------

    %PYTHON% %SCRIPT_TEST_DISTRIBUTION% "!FILE_PHILOX!" "!FILE_RANDOMORG!"
)
