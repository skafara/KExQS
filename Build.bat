@echo off

if not exist obj mkdir obj
if not exist bin mkdir bin

del /q /s "obj\*"
del /q /s "bin\*"

@echo on
cl /std:c++20 /EHsc /LD /Foobj\ /Febin\ESimulator.dll /I src\cpp\include src\cpp\lib\**.cpp /I %OPENCL_SDK_PATH%\include /D CL_HPP_TARGET_OPENCL_VERSION=200 /D OPENCL_KERNELS_PATH=\"%OPENCL_KERNELS_PATH%\" /link /LIBPATH:%OPENCL_SDK_PATH%\lib OpenCL.lib
xcopy /Y /E /I src\cpp\kernels bin\%OPENCL_KERNELS_PATH%
fpc -Px86_64 -dRUN_EXTERNAL -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
