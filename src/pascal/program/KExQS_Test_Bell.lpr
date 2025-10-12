program KEQS_Test01;

{$MODE ObjFPC}
{$H+}

uses SysUtils, KQS.Complex, KQS.Algebra, KQS.Circuit, KQS.Simulator, KQS.Random;

procedure TestBell;
var
  I: Integer;
  R1: TQuantumRegister;
  Sim: TQuantumSimulator;
begin
  WriteLn;
  WriteLn('*** BELL STATE TEST ***');

  R1 := TQuantumRegister.Create(2);

  R1.Hadamard(0);
  R1.ControlledNot(0, 1);

  WriteLn('Error = ', R1.Error);
  WriteLn(R1.StateVectorAsString);

  Sim := TQuantumSimulator.Create(R1);
  Sim.Run(1024);

  { Expected results: 00 and 11 with approximately the same counts }
  { Experimental results: 00, 01, 10, 11 with approximately the same counts }
  { This is due to the fact that qubits are measured independently one by one, !without collapsing! }
  { When collapsing is used, only expected states are obtained. }
  for I := 0 to Sim.States - 1 do
    WriteLn('State ', BinToStr(I, R1.Qubits), ' measured ', Sim.GetStateCount(I), ' times.');

  Sim.Free;
  R1.Free;
end;

begin
  TestBell;
  ReadLn;
end.

