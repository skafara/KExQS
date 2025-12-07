@echo off

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
    /D CL_HPP_TARGET_OPENCL_VERSION=200 ^
    /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" ^
    /link ^
    /LIBPATH:%TBB_LIB% ^
    /LIBPATH:%OPENCL_LIB% ^
    tbb12.lib ^
    OpenCL.lib

REM Copy required runtime files

copy %TBB_REDIST%\tbb12.dll bin
xcopy /Y /E /I src\cpp\kernels bin\%OPENCL_KERNELS_PATH%

REM Build Pascal Application

fpc -Px86_64 -dRUN_EXTERNAL -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
