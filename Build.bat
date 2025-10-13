@echo off

if not exist obj mkdir obj
if not exist bin mkdir bin

del /q /s "obj\*"
del /q /s "bin\*"

@echo on

cl /std:c++20 /LD /Foobj\ /Febin\ESimulator.dll /I src\cpp\include src\cpp\lib\KQS.Simulator.cpp
:: fpc -Px86_64 -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
fpc -Px86_64 -dRUN_EXTERNAL -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
