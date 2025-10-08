program KEQS_Test01;

{$MODE ObjFPC}
{$H+}

uses SysUtils, KQS.Complex, KQS.Algebra, KQS.Circuit, KQS.Simulator, KQS.Random;

procedure TestComplexNumbers;
var
  A, B, C, D: Complex;
begin
  WriteLn;
  WriteLn('*** COMPLEX NUMBER TESTS ***');

  A := Cplx(3, 2);
  B := Complex([1, 7]);

  WriteLn('A = ', string(A));
  WriteLn('B = ', string(B));

  C := A + B;
  WriteLn('C (A + B) = ', string(C));

  C := A - B;
  WriteLn('C (A - B) = ', string(C));

  C := A * B;
  WriteLn('C (A * B) = ', string(C));

  C := A / B;
  WriteLn('C (A / B) = ', string(C));

  D := B;
  WriteLn('D = ', string(D));

  WriteLn('|A| = ', CMag(A));
  WriteLn('CConj(A) = ', string(Conj(A)));
end;

procedure TestPCGRandomGenerator;
var
  R: TPCG64Random;
  I: Integer;
begin
  WriteLn;
  WriteLn('*** PCG RANDOM GENERATOR TESTS ***');

  PCG_SetSeq_128_SRandom_R(R, PCG_128BIT_CONSTANT(0, UInt64(Now)),
    PCG_128BIT_CONSTANT(0, UInt64(Now)));

  for I := 1 to 100 do
    WriteLn(PCG_SetSeq_128_XSL_RR_64_Random_R(R));
end;

procedure TestCPURandomGenerator;
var
  I: Integer;
begin
  WriteLn;
  WriteLn('*** CPU (RDRAND) RANDOM GENERATOR TESTS ***');

  for I := 1 to 100 do
    WriteLn(RDRAND_UNorm);
end;

procedure TestAlgebra;
var
  V1, V2: TComplexVector;
  M1: TComplexMatrix;
begin
  WriteLn;
  WriteLn('*** COMPLEX ALGEBRA TESTS ***');

  WriteLn('2^5 = ', TwoPower(5));

  V1 := TComplexVector.Create(5);
  V1.SetComponent(0, Cplx(2.0, -3.0));
  V1.SetComponent(4, Cplx(3.0, 2.0));

  WriteLn('V1 = ', V1.ToString);

  V2 := TComplexVector.Copy(V1);
  WriteLn('V2 = ', V2.ToString);

  V1.Free;
  V2.Free;

  M1 := TComplexMatrix.Create(4, 4);
  M1.Print;
  M1.Free;

  WriteLn('e^(2+5i) = ', string(CEulerPower(Cplx(2, 5))));
end;

procedure TestQuantumRegister;
var
  R1: TQuantumRegister;
begin
  WriteLn;
  WriteLn('*** QUBIT REGISTER TESTS ***');

  R1 := TQuantumRegister.Create(2);
  R1.Initialize(StrToBin('10'), Cplx(0, 1));

  WriteLn(R1.StateVectorAsString);

  R1.Free;
end;

procedure TestQuantumGates;
var
  I: Integer;
  R1: TQuantumRegister;
  Sim: TQuantumSimulator;
begin
  WriteLn;
  WriteLn('*** QUANTUM GATES TESTS ***');

  R1 := TQuantumRegister.Create(3);

  R1.Hadamard(0);
  R1.PauliX(1);
  R1.PauliX(2);
  R1.ControlledNot(1, 0);
  R1.Swap(0, 2);
  R1.PiOverEight(0);
  R1.Phase(kqsPi / 3.0, 1);
  R1.Toffoli(0, 1, 2);
  R1.ControlledPhase(kqsPi / 4.0, 0, 2);  // CP makes the error

  WriteLn('Error = ', R1.Error);
  WriteLn(R1.StateVectorAsString);

  Sim := TQuantumSimulator.Create(R1);
  Sim.Run(1024);

  for I := 0 to Sim.States - 1 do
    WriteLn('State ', BinToStr(I, R1.Qubits), ' measured ', Sim.GetStateCount(I), ' times.');

  Sim.Free;
  R1.Free;
end;

begin
  TestComplexNumbers;
  TestPCGRandomGenerator;
  TestCPURandomGenerator;
  TestAlgebra;

  TestQuantumRegister;
  TestQuantumGates;

  ReadLn;
end.

