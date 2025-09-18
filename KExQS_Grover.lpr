program KExQS_Grover;

uses
  SysUtils, KQS.Complex, KQS.Algebra, KQS.Circuit, KQS.Simulator;

var
  R: TQuantumRegister;
  S: TQuantumSimulator;

  I: Byte;
  J: Cardinal;

begin
  R := TQuantumRegister.Create(4);

  { Initialisation }
  for I := 0 to 3 do R.Hadamard(I);

  { Oracle for 0000 }
  for I := 0 to 3 do R.PauliX(I);

  R.ControlledPhase(kqsPi / 4.0, 0, 3);
  R.ControlledNot(0, 1);
  R.ControlledPhase(-kqsPi / 4.0, 1, 3);
  R.ControlledNot(0, 1);
  R.ControlledPhase(kqsPi / 4.0, 1, 3);
  R.ControlledNot(1, 2);
  R.ControlledPhase(-kqsPi / 4.0, 2, 3);
  R.ControlledNot(0, 2);
  R.ControlledPhase(kqsPi / 4.0, 2, 3);
  R.ControlledNot(1, 2);
  R.ControlledPhase(-kqsPi / 4.0, 2, 3);
  R.ControlledNot(0, 2);
  R.ControlledPhase(kqsPi / 4.0, 2, 3);

  for I := 0 to 3 do R.PauliX(I);

  { Amplification }
  for I := 0 to 3 do R.Hadamard(I);
  for I := 0 to 3 do R.PauliX(I);

  R.ControlledPhase(kqsPi / 4.0, 0, 3);
  R.ControlledNot(0, 1);
  R.ControlledPhase(-kqsPi / 4.0, 1, 3);
  R.ControlledNot(0, 1);
  R.ControlledPhase(kqsPi / 4.0, 1, 3);
  R.ControlledNot(1, 2);
  R.ControlledPhase(-kqsPi / 4.0, 2, 3);
  R.ControlledNot(0, 2);
  R.ControlledPhase(kqsPi / 4.0, 2, 3);
  R.ControlledNot(1, 2);
  R.ControlledPhase(-kqsPi / 4.0, 2, 3);
  R.ControlledNot(0, 2);
  R.ControlledPhase(kqsPi / 4.0, 2, 3);

  for I := 0 to 3 do R.PauliX(I);
  for I := 0 to 3 do R.Hadamard(I);

  { Simulation }
  S := TQuantumSimulator.Create(R);

  S.Run(10000);
  for J := 0 to S.States - 1 do
    WriteLn('State ', BinToStr(J, R.Qubits), ' measured ',
      S.GetStateCount(J), ' times.');

  S.Free;
  R.Free;

  ReadLn;
end.

