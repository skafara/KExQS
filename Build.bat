@echo off

if not exist obj mkdir obj
if not exist bin mkdir bin

@echo on

cl /LD src\cpp\Test_External.c /Foobj\ /Febin\Test_External.dll
fpc -Px86_64 -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test_External.lpr
