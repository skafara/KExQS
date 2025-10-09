@echo off

if not exist obj mkdir obj
if not exist bin mkdir bin

@echo on

fpc -Px86_64 -FEbin -FUobj -Foobj -Fusrc\pascal\lib src\pascal\program\KExQS_Test01.lpr
