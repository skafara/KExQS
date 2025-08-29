program KEQS_Test01;

uses SysUtils, KQS.Complex, KQS.Algebra, KQS.Circuit;

procedure TestComplexNumbers;
var
  A, B, C, D: Complex;
begin
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

procedure TestAlgebra;
var
  V1, V2: TComplexVector;
  M1: TComplexMatrix;
begin
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

function GetNthBit(State, Qubit: Cardinal): string;
var
  Pos, Bit: Cardinal;
begin
  Pos := 4 - Qubit - 1;
  Bit := (State shr Pos) and 1;
  Result := IntToStr(Bit);
end;

procedure TestQuantumRegister;
var
  R1: TQuantumRegister;
begin
  WriteLn('*** QUBIT REGISTER TESTS ***');

  R1 := TQuantumRegister.Create(2);

//  WriteLn('01010 = ', StrToBin('01010'));
  R1.Initialize(StrToBin('10'), Cplx(0, 1));

  WriteLn(R1.StateVectorAsString);

  WriteLn('GetNthBit: ');
  WriteLn(GetNthBit(10, 2));

  R1.Free;
end;

procedure TestQuantumGates;
var
  R1: TQuantumRegister;
  QB: Boolean;
  I, N0, N1: Integer;
begin
  WriteLn('*** QUANTUM GATES TESTS ***');

  R1 := TQuantumRegister.Create(3);

  //WriteLn('Hadamard Gate Matrix:');
  //H.Matrix.Print;

  R1.Hadamard(0);
  R1.PauliX(1);
  R1.PauliX(2);
  R1.ControlledNot(1, 0);
  R1.Swap(0, 2);
  R1.PiOverEight(0);
  R1.Phase(kqsPi / 3.0, 1);
  R1.Toffoli(0, 1, 2);

  WriteLn('Error = ', R1.Error);
  WriteLn(R1.StateVectorAsString);

  N0 := 0;
  N1 := 0;
  for I := 1 to 1024 do
  begin
    if I < 1024 then QB := R1.Measure(0, False) else QB := R1.Measure(0);
    if QB then Inc(N1) else Inc(N0);
  end;
  WriteLn('Measured qubit 0 -> "0" in ', N0, '/1024 (',
    N0 / 1024 * 100:1:2, '%), "1" in ', N1, '/1024 (', N1 / 1024 * 100:1:2, '%)');
  WriteLn('Post-measurement state:');
  WriteLn(R1.StateVectorAsString);

  R1.Free;
end;

begin
  TestComplexNumbers;
  TestAlgebra;

  TestQuantumRegister;
  TestQuantumGates;

  ReadLn;
end.

