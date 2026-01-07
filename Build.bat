@echo off

REM =====================================================
REM Default build configuration
REM =====================================================

set EXECUTION_POLICY=Accelerated
set PRNG_ALGORITHM=Philox

REM =====================================================
REM Override from CLI
REM =====================================================

if not "%~1"=="" set EXECUTION_POLICY=%~1
if not "%~2"=="" set PRNG_ALGORITHM=%~2

REM =====================================================
REM Validate EXECUTION_POLICY
REM =====================================================

if /I "%EXECUTION_POLICY%"=="Sequential" (
    REM ok
) else if /I "%EXECUTION_POLICY%"=="Parallel" (
    REM ok
) else if /I "%EXECUTION_POLICY%"=="Accelerated" (
    REM ok
) else (
    echo ERROR: Invalid EXECUTION_POLICY "%EXECUTION_POLICY%"
    echo Valid values: Sequential ^| Parallel ^| Accelerated
    exit /b 1
)

REM =====================================================
REM Validate PRNG_ALGORITHM
REM =====================================================

if /I "%PRNG_ALGORITHM%"=="Philox" (
    REM ok
) else if /I "%PRNG_ALGORITHM%"=="MT19937" (
    REM ok
) else if /I "%PRNG_ALGORITHM%"=="RandomOrg" (
    REM ok
) else (
    echo ERROR: Invalid PRNG_ALGORITHM "%PRNG_ALGORITHM%"
    echo Valid values: Philox ^| MT19937 ^| RandomOrg
    exit /b 1
)

REM =====================================================
REM Build configuration
REM =====================================================

echo ========================================
echo Build configuration:
echo   EXECUTION_POLICY = %EXECUTION_POLICY%
echo   PRNG_ALGORITHM   = %PRNG_ALGORITHM%
echo ========================================


REM Ensure obj/bin directories exist

if not exist obj mkdir obj
if not exist bin mkdir bin

REM Delete previous build files

del /q /s "obj\*"
del /q /s "bin\*"

REM Build C++ DLL

cl  /std:c++20 /EHsc /LD /Foobj\ /Febin\ESimulator.dll ^
    /I src\cpp\include ^
    src\cpp\lib\**.cpp ^
    /I %TBB_INCLUDE% ^
    /I %OPENCL_INCLUDE% ^
    /O2 ^
    /Ot ^
    /Ob2 ^
    /fp:precise ^
    /D CL_HPP_TARGET_OPENCL_VERSION=300 ^
    /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
    /D RANDOMORG_FILES_PATH=\"%RANDOMORG_FILES_PATH%\" ^
    /D EXECUTION_POLICY=%EXECUTION_POLICY% ^
    /D PRNG_ALGORITHM=%PRNG_ALGORITHM% ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    /OPT:REF ^
    tbb12.lib ^
    OpenCL.lib

REM Copy required runtime files

copy %TBB_REDIST%\tbb12.dll bin
xcopy /Y /E /I src\cpp\kernels bin\%OPENCL_KERNELS_PATH%

REM Build Pascal Application

fpc -Px86_64 -dRUN_EXTERNAL -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
