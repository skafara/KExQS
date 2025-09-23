unit KQS.Circuit;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Circuit
         Version 1.0

         Copyright (c) Kamil Ekstein, 2025
         All rights reserved.

         Code written by: Kamil Ekstein
         Target compiler: FPC 3.2.2
         Last update on:  11-Sep-2025
         Last update by:  KE
  ______________________________________________________________________________
}

{$MODE ObjFPC}
{$H+}

interface

uses
  Types, Classes, SysUtils, Math, KQS.Complex, KQS.Algebra, KQS.Random;

const
  kqsMaxSimulatedQubits = 20;

  errOK = 0;
  errEmptyRegister = 1;
  errOutOfMemory = 2;
  errTaskTooLarge = 3;
  errIncorrectSize = 4;
  errNoTargetQubits = 5;

type
  TQuantumLogicGate = class(TObject)
  private
    FDimension: Cardinal;
    FMatrix: TComplexMatrix;
  public
    property Dimension: Cardinal read FDimension;
    property Matrix: TComplexMatrix read FMatrix;

    constructor Create(ADimension: Cardinal);
    destructor Destroy; override;

    class function PauliX: TQuantumLogicGate;
    class function PauliY: TQuantumLogicGate;
    class function PauliZ: TQuantumLogicGate;
    class function Hadamard: TQuantumLogicGate;
    class function Phase(APhase: Real): TQuantumLogicGate;
    class function ControlledPhase(APhase: Real): TQuantumLogicGate;
    class function PiOverEight: TQuantumLogicGate;
    class function ControlledNot: TQuantumLogicGate;
    class function ControlledZ: TQuantumLogicGate;
    class function Swap: TQuantumLogicGate;
    class function Toffoli: TQuantumLogicGate;

    class function MakeControlled(AGate: TQuantumLogicGate;
      AControlQubits: TByteDynArray): TQuantumLogicGate;
  end;

  TQuantumRegister = class(TObject)
  private
    FError: Byte;
    FNumQubits: Byte;
    FNumStates: Cardinal;
    FStateVector: TComplexVector; // vector of probability amplitudes
  private
    procedure InitializeToZeroState;
    procedure Normalize;

    procedure ApplyOneQubitGate(AGate: TQuantumLogicGate; ATargetQubit: Byte);
    procedure ApplyTwoQubitGate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
    procedure ApplyKQubitGate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
    procedure ApplyKQubitControlledGate(AGate: TQuantumLogicGate;
      AControlQubits, ATargetQubits: TByteDynArray);
  public
    property Error: Byte read FError;
    property Qubits: Byte read FNumQubits;

    constructor Create(ANumberOfQubits: Byte);
    destructor Destroy; override;

    function Amplitude(AState: Cardinal): Complex;
    function Probability(AState: Cardinal): Real;
    function StateVectorAsString: string;

    procedure Initialize(AState: Cardinal; AAmplitude: Complex);
    function Measure(ATargetQubit: Byte; ACollapse: Boolean = True): Boolean;

    procedure Hadamard(ATargetQubit: Byte);
    procedure PauliX(ATargetQubit: Byte);
    procedure PauliY(ATargetQubit: Byte);
    procedure PauliZ(ATargetQubit: Byte);
    procedure PiOverEight(ATargetQubit: Byte);
    procedure Phase(APhase: Real; ATargetQubit: Byte);
    procedure ControlledPhase(APhase: Real; AControlQubit, ATargetQubit: Byte);
    procedure Swap(ATargetQubit1, ATargetQubit2: Byte);
    procedure ControlledNot(AControlQubit, ATargetQubit: Byte);
    procedure ControlledZ(AControlQubit, ATargetQubit: Byte);
    procedure Toffoli(AControlQubit1, AControlQubit2, ATargetQubit: Byte);

    procedure Gate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
  end;

var
  PCGRandom: TPCG64Random;

implementation

{ ______________________________________________________________________________

         TQuantumLogicGate
         CLASS IMPLEMENTATION
  ______________________________________________________________________________
}

constructor TQuantumLogicGate.Create(ADimension: Cardinal);
begin
  inherited Create;

  FDimension := ADimension;

  FMatrix := TComplexMatrix.Create(ADimension, ADimension);
  if (FMatrix.Rows = 0) or
     (FMatrix.Columns = 0) then
  begin
    FDimension := 0;
  end;
end;

destructor TQuantumLogicGate.Destroy;
begin
  if FDimension > 0 then FMatrix.Free;

  inherited Destroy;
end;

class function TQuantumLogicGate.Hadamard: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(2);
  Result.Matrix.SetElement(0, 0, Complex([kqs1OverSqrt2, 0.0]));
  Result.Matrix.SetElement(0, 1, Complex([kqs1OverSqrt2, 0.0]));
  Result.Matrix.SetElement(1, 0, Complex([kqs1OverSqrt2, 0.0]));
  Result.Matrix.SetElement(1, 1, Complex([-kqs1OverSqrt2, 0.0]));
end;

class function TQuantumLogicGate.PauliX: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(2);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([0.0, 0.0]));
    SetElement(0, 1, Complex([1.0, 0.0]));
    SetElement(1, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, Complex([0.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.PauliY: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(2);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([0.0, 0.0]));
    SetElement(0, 1, Complex([0.0, -1.0]));
    SetElement(1, 0, Complex([0.0, 1.0]));
    SetElement(1, 1, Complex([0.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.PauliZ: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(2);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(0, 1, Complex([0.0, 0.0]));
    SetElement(1, 0, Complex([0.0, 0.0]));
    SetElement(1, 1, Complex([-1.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.Phase(APhase: Real): TQuantumLogicGate;
var
  Ph: Real;
begin
  if (APhase >= 0.0) and (APhase < 2.0 * kqsPi) then Ph := APhase
  else Ph := FMod(APhase, 2.0 * kqsPi);

  Result := TQuantumLogicGate.Create(2);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, CEulerPower(Complex([0.0, Ph])));
  end;
end;

class function TQuantumLogicGate.ControlledPhase(APhase: Real): TQuantumLogicGate;
var
  Ph: Real;
begin
  if (APhase >= 0.0) and (APhase < 2.0 * kqsPi) then Ph := APhase
  else Ph := FMod(APhase, 2.0 * kqsPi);

  Result := TQuantumLogicGate.Create(4);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, Complex([1.0, 0.0]));
    SetElement(2, 2, Complex([1.0, 0.0]));
    SetElement(3, 3, CEulerPower(Complex([0.0, Ph])));
  end;
end;

class function TQuantumLogicGate.PiOverEight: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(2);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, CEulerPower(Complex([0.0, kqsPi / 4.0])));
  end;
end;

class function TQuantumLogicGate.ControlledNot: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(4);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, Complex([1.0, 0.0]));
    SetElement(2, 3, Complex([1.0, 0.0]));
    SetElement(3, 2, Complex([1.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.ControlledZ: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(4);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, Complex([1.0, 0.0]));
    SetElement(2, 2, Complex([1.0, 0.0]));
    SetElement(3, 3, Complex([-1.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.Swap: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(4);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 2, Complex([1.0, 0.0]));
    SetElement(2, 1, Complex([1.0, 0.0]));
    SetElement(3, 3, Complex([1.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.Toffoli: TQuantumLogicGate;
begin
  Result := TQuantumLogicGate.Create(8);
  with Result.Matrix do
  begin
    SetElement(0, 0, Complex([1.0, 0.0]));
    SetElement(1, 1, Complex([1.0, 0.0]));
    SetElement(2, 2, Complex([1.0, 0.0]));
    SetElement(3, 3, Complex([1.0, 0.0]));
    SetElement(4, 4, Complex([1.0, 0.0]));
    SetElement(5, 5, Complex([1.0, 0.0]));
    SetElement(6, 7, Complex([1.0, 0.0]));
    SetElement(7, 6, Complex([1.0, 0.0]));
  end;
end;

class function TQuantumLogicGate.MakeControlled(AGate: TQuantumLogicGate;
  AControlQubits: TByteDynArray): TQuantumLogicGate;
var
  Dim, BlkSize: Cardinal;
  LC: Byte;
  I, J: Cardinal;
begin
  LC := Length(AControlQubits);
  BlkSize := AGate.Dimension;
  Dim := (1 shl LC) * BlkSize;

  Result := TQuantumLogicGate.Create(Dim);

  { set identities to all states except where all control qubits are 1s }
  for I := 0 to Dim - 1 do
    Result.Matrix.SetElement(I, I, Complex([1.0, 0.0]));

  { overwrite bottom-right block with the gate matrix }
  for I := 0 to BlkSize - 1 do
    for J := 0 to BlkSize - 1 do
      Result.Matrix.SetElement(Dim - BlkSize + I, Dim - BlkSize + J,
        AGate.Matrix.GetElement(I, J));
end;

{ ______________________________________________________________________________

         TQuantumRegister
         CLASS IMPLEMENTATION
  ______________________________________________________________________________
}

constructor TQuantumRegister.Create(ANumberOfQubits: Byte);
begin
  inherited Create;

  { check if the number of simlated bits is sane }
  if (ANumberOfQubits > kqsMaxSimulatedQubits) or
     (ANumberOfQubits = 0) then
  begin
    FNumQubits := 0;
    FNumStates := 0;
    Exit;
  end;

  { initialize the state variables }
  FNumQubits := ANumberOfQubits;
  FNumStates := TwoPower(FNumQubits);
  FStateVector := TComplexVector.Create(FNumStates);
  InitializeToZeroState; { set the probability of the state |00...00> to 1 }

  { check if the memory allocation was successful }
  if FStateVector.Size = 0 then
  begin
    FNumQubits := 0;
    FNumStates := 0;
    FreeAndNil(FStateVector);
  end;
end;

destructor TQuantumRegister.Destroy;
begin
  inherited Destroy;
end;

procedure TQuantumRegister.InitializeToZeroState;
begin
  FStateVector.Components^[0] := Complex([1.0, 0.0]);
end;

function TQuantumRegister.Amplitude(AState: Cardinal): Complex;
begin
  if AState <= FNumStates then
  begin
    Result.Re := FStateVector.Components^[AState].Re;
    Result.Im := FStateVector.Components^[AState].Im;
  end
  else
  begin
    Result.Re := 0.0;
    Result.Im := 0.0;
  end;
end;

function TQuantumRegister.Probability(AState: Cardinal): Real;
var
  C: Complex;
begin
  C := Amplitude(AState);
  Result := C.Re * C.Re + C.Im * C.Im;
end;

function TQuantumRegister.StateVectorAsString: string;
var
  I: Cardinal;
  SS: string;
begin
  if FNumStates = 0 then Exit;

  Result := '';
  for I := 0 to FNumStates - 1 do
  begin
    SS := '|' + BinToStr(I, FNumQubits) + '>';
    Result := Result +
      SS + ': ' + string(FStateVector.Components^[I]) +
      '; p(' + SS + ') = ' +
      FloatToStrF(Probability(I), ffFixed, 2, 2) +
      #10#13;
  end;
end;

procedure TQuantumRegister.Initialize(AState: Cardinal; AAmplitude: Complex);
begin
  { check whether the state exists }
  if AState > FNumStates then Exit;

  { set the state amplitude accordingly }
  FStateVector.Components^[AState] := AAmplitude;
end;

procedure TQuantumRegister.Normalize;
var
  I: Cardinal;
  Sum, Factor: Real;
begin
  { check whether there is something to normalize }
  if FNumStates = 0 then Exit;

  Sum := 0.0;
  for I := 0 to FNumStates - 1 do
    Sum := Sum + CAbsSquared(FStateVector.Components^[I]);

  if Sum <= 0 then Exit;

  Factor := 1.0 / Sqrt(Sum);
  for I := 0 to FNumStates - 1 do
    FStateVector.Components^[I] := CScale(FStateVector.Components^[I], Factor);
end;

procedure TQuantumRegister.ApplyOneQubitGate(AGate: TQuantumLogicGate; ATargetQubit: Byte);
var
  N, Step, Base, I, J: Cardinal;
  A, B, A_New, B_New: Complex;
begin
  N := 1 shl FNumQubits;
  Step := 1 shl ATargetQubit;
  Base := 0;

  while Base < N do
  begin
    for I := 0 to Step - 1 do
    begin
      J := Base + I; { index where the target bit = 0 }
      A := FStateVector.Components^[J];
      B := FStateVector.Components^[J + Step]; { index where the target bit = 1 }
      A_New := AGate.Matrix.GetElement(0, 0) * A + AGate.Matrix.GetElement(0, 1) * B;
      B_New := AGate.Matrix.GetElement(1, 0) * A + AGate.Matrix.GetElement(1, 1) * B;
      FStateVector.Components^[J] := A_New;
      FStateVector.Components^[J + Step] := B_New;
    end;

    Inc(Base, Step * 2);
  end;
end;

procedure TQuantumRegister.ApplyTwoQubitGate(AGate: TQuantumLogicGate;
  ATargetQubits: TByteDynArray);
var
  N, Mask0, Mask1: Cardinal;
  I00, I01, I10, I11: Cardinal;
  A00, A01, A10, A11: Complex;
  NA00, NA01, NA10, NA11: Complex;
begin
  { check the initial conditions sanity }
  if FNumStates = 0 then Exit;
  if Length(ATargetQubits) <> 2 then Exit;
  if ATargetQubits[0] = ATargetQubits[1] then Exit;

  Mask0 := 1 shl ATargetQubits[0];
  Mask1 := 1 shl ATargetQubits[1];

  for N := 0 to FNumStates - 1 do
  begin
    { skip the loop body if either target bit is set }
    if ((N and Mask0) <> 0) or ((N and Mask1) <> 0) then Continue;

    { compute the indices }
    I00 := N;
    I01 := N or Mask1;
    I10 := N or Mask0;
    I11 := N or Mask0 or Mask1;

    { get the original amplitudes }
    A00 := FStateVector.Components^[I00];
    A01 := FStateVector.Components^[I01];
    A10 := FStateVector.Components^[I10];
    A11 := FStateVector.Components^[I11];

    { mutliply by the ApplyKQubitGate matrix }
    with AGate.Matrix do
    begin
      NA00 := GetElement(0, 0) * A00 + GetElement(0, 1) * A01 +
              GetElement(0, 2) * A10 + GetElement(0, 3) * A11;
      NA01 := GetElement(1, 0) * A00 + GetElement(1, 1) * A01 +
              GetElement(1, 2) * A10 + GetElement(1, 3) * A11;
      NA10 := GetElement(2, 0) * A00 + GetElement(2, 1) * A01 +
              GetElement(2, 2) * A10 + GetElement(2, 3) * A11;
      NA11 := GetElement(3, 0) * A00 + GetElement(3, 1) * A01 +
              GetElement(3, 2) * A10 + GetElement(3, 3) * A11;
    end;

    { store the new amplitudes back to the state vector }
    FStateVector.Components^[I00] := NA00;
    FStateVector.Components^[I01] := NA01;
    FStateVector.Components^[I10] := NA10;
    FStateVector.Components^[I11] := NA11;
  end;
end;

procedure TQuantumRegister.ApplyKQubitGate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
var
  State, NewState: Cardinal;
  I, J, K, L, R: Cardinal;
  QbPos, QbLen: Byte;
  Amplitudes: TComplexVector;
  TempBit: Cardinal;
  TempPos: Byte;
begin
  { check if the initial conditions are sane }
  if FNumStates = 0 then
  begin
    FError := errEmptyRegister;
    Exit;
  end;

  { check the number of target qubits }
  QbLen := Length(ATargetQubits);
  if (QbLen > kqsMaxSimulatedQubits) or
     (QbLen > FNumQubits) then
  begin
    FError := errTaskTooLarge;
    Exit;
  end;

  { check if the ApplyKQubitGate size corresponds to the number of qubits }
  if AGate.Dimension <> TwoPower(QbLen) then
  begin
    FError := errIncorrectSize;
    Exit;
  end;

  { copy the current state }
  Amplitudes := TComplexVector.Copy(FStateVector);

  { apply the ApplyKQubitGate }
  for State := 0 to FNumStates - 1  do
  begin
    if (Amplitudes.Components^[State].Re <> 0.0) or
       (Amplitudes.Components^[State].Im <> 0.0) then
    begin
      { get the R (row, col) in the ApplyKQubitGate matrix }
      R := 0;
      for I := 0 to Length(ATargetQubits) - 1 do
      begin
        QbPos := ATargetQubits[I];
        if (State and (1 shl QbPos)) > 0 then
        begin
          R := R or (1 shl I);
        end;
      end;

      FStateVector.Components^[State] := FStateVector.Components^[State] -
        (Complex([1.0, 0.0]) - AGate.Matrix.Elements.Components^[R * AGate.Matrix.Columns + R]) *
          Amplitudes.Components^[State];

      J := 0;
      for K := 0 to TwoPower(QbLen) - 1 do
      begin
        if J <> R then
        begin
          NewState := State;
          I := 0;

          for L := QbLen - 1 downto 0 do
          begin
            if I >= QbLen then Break;

            TempBit := (K shr L) and 1;
            TempPos := FNumQubits - ATargetQubits[I] - 1;

            NewState := NewState or (TempBit shl TempPos);

            Inc(I);
          end;

          FStateVector.Components^[NewState] := FStateVector.Components^[NewState] +
            AGate.Matrix.Elements.Components^[J * AGate.Matrix.Columns + R] *
              Amplitudes.Components^[State];
        end;

        Inc(J);
      end;
    end;
  end;

  Amplitudes.Free;
end;

procedure TQuantumRegister.ApplyKQubitControlledGate(AGate: TQuantumLogicGate;
  AControlQubits, ATargetQubits: TByteDynArray);
var
  I, LC, LT: Byte;
  CGate: TQuantumLogicGate;
  AffectedQubits: TByteDynArray;
begin
  { check the sanity of the initial conditions }
  LC := Length(AControlQubits);
  LT := Length(ATargetQubits);
  if (LC = 0) or (LT = 0) then
  begin
    FError := errNoTargetQubits;
    Exit;
  end;

  { build the controlled version of the gate matrix }
  CGate := TQuantumLogicGate.MakeControlled(AGate, AControlQubits);

  { merge control and target AffectedQubits }
  SetLength(AffectedQubits, LC + LT);
  for I := 0 to High(AControlQubits) do
    AffectedQubits[I] := AControlQubits[I];
  for I := 0 to High(ATargetQubits) do
    AffectedQubits[LC + I] := ATargetQubits[I];

  { apply the gate }
  ApplyKQubitGate(CGate, AffectedQubits);

  { cleanup }
  CGate.Free;
end;

(*
procedure TQuantumRegister.ApplyKQubitGate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
var
  State, NewState: Cardinal;
  I, K: Cardinal;
  QbLen: Byte;
  R, TempBit: Cardinal;
  QbPos: Byte;
  Amplitudes: TComplexVector;
begin
  { sanity checks }
  if FNumStates = 0 then
  begin
    FError := errEmptyRegister;
    Exit;
  end;

  QbLen := Length(ATargetQubits);
  if (QbLen > kqsMaxSimulatedQubits) or (QbLen > FNumQubits) then
  begin
    FError := errTaskTooLarge;
    Exit;
  end;

  if AGate.Dimension <> TwoPower(QbLen) then
  begin
    FError := errIncorrectSize;
    Exit;
  end;

  { copy the current state }
  Amplitudes := TComplexVector.Copy(FStateVector);

  { apply the K-qubit gate }
  for State := 0 to FNumStates - 1 do
  begin
    if (Amplitudes.Components^[State].Re <> 0.0) or
       (Amplitudes.Components^[State].Im <> 0.0) then
    begin
      { determine row index R (local state of target qubits) }
      R := 0;
      for I := 0 to QbLen - 1 do
      begin
        QbPos := ATargetQubits[I];
        if (State and (1 shl QbPos)) <> 0 then
          R := R or (1 shl I);  { local numbering }
      end;

      { distribute amplitudes to all possible new target states }
      for K := 0 to TwoPower(QbLen) - 1 do
      begin
        NewState := State;

        { clear target bits }
        for I := 0 to QbLen - 1 do
        begin
          QbPos := ATargetQubits[I];
          NewState := NewState and not (1 shl QbPos);
        end;

        { set target bits according to K }
        for I := 0 to QbLen - 1 do
        begin
          QbPos := ATargetQubits[I];
          TempBit := (K shr I) and 1;
          if TempBit <> 0 then
            NewState := NewState or (1 shl QbPos);
        end;

        { add contribution from matrix element }
        FStateVector.Components^[NewState] :=
          FStateVector.Components^[NewState] +
          AGate.Matrix.Elements.Components^[K * AGate.Matrix.Columns + R] *
          Amplitudes.Components^[State];
      end;
    end;
  end;

  Amplitudes.Free;
end;
*)

function TQuantumRegister.Measure(ATargetQubit: Byte; ACollapse: Boolean = True): Boolean;
var
  I: Cardinal;
  Prob0, R: Real;
begin
  Prob0 := 0.0;
  for I := 0 to FNumStates - 1 do
    if ((I shr ATargetQubit) and 1) = 0 then
      Prob0 := Prob0 + CAbsSquared(FStateVector.Components^[I]);

  { collapse the states }
  //R := Random;
  //R := RDRAND_UNorm;
  R := PCGRandom_UNorm(PCGRandom);

  if R < Prob0 then
  begin
    { result 0: zero amplitudes with bit = 1 }
    if ACollapse then
    begin
      for I := 0 to FNumStates - 1 do
        if ((I shr ATargetQubit) and 1) = 1 then
          FStateVector.Components^[I] := Complex([0.0, 0.0]);
      Normalize;
    end;

    Result := False;
  end
  else
  begin
    { result 1: zero amplitudes with bit = 0 }
    if ACollapse then
    begin
      for I := 0 to FNumStates - 1 do
        if ((I shr ATargetQubit) and 1) = 0 then
          FStateVector.Components^[I] := Complex([0.0, 0.0]);
      Normalize;
    end;

    Result := True;
  end;
end;

{ ______________________________________________________________________________

         QUANTUM LOGIC GATES
  ______________________________________________________________________________
}
{ qiskit qc.h() }
procedure TQuantumRegister.Hadamard(ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Hadamard (H) gate and apply it }
  H := TQuantumLogicGate.Hadamard;
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.x() }
procedure TQuantumRegister.PauliX(ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Pauli-X (X) gate and apply it }
  H := TQuantumLogicGate.PauliX;
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.y() }
procedure TQuantumRegister.PauliY(ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Pauli-Y (Y) gate and apply it }
  H := TQuantumLogicGate.PauliY;
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.z() }
procedure TQuantumRegister.PauliZ(ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Pauli-Z (Z) gate and apply it }
  H := TQuantumLogicGate.PauliZ;
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.t() }
procedure TQuantumRegister.PiOverEight(ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Pi-Over-Eight (T) gate and apply it }
  H := TQuantumLogicGate.PiOverEight;
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.swap() }
procedure TQuantumRegister.Swap(ATargetQubit1, ATargetQubit2: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit1 >= FNumQubits) or
     (ATargetQubit2 >= FNumQubits) then Exit;

  { create the Swap gate (SWAP) and apply it }
  H := TQuantumLogicGate.Swap;
  ApplyTwoQubitGate(H, [ATargetQubit1, ATargetQubit2]);
  H.Free;
end;

{ qiskit qc.p() }
procedure TQuantumRegister.Phase(APhase: Real; ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Phase (P) gate and apply it }
  H := TQuantumLogicGate.Phase(APhase);
  ApplyOneQubitGate(H, ATargetQubit);
  H.Free;
end;

{ qiskit qc.cp() }
procedure TQuantumRegister.ControlledPhase(APhase: Real; AControlQubit, ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) then Exit;

  { create the Controlled Phase (CP) gate and apply it }
  H := TQuantumLogicGate.ControlledPhase(APhase);
  ApplyTwoQubitGate(H, [AControlQubit, ATargetQubit]);
  //ApplyControlledKQubitGate(H, [AControlQubit], [ATargetQubit]);
  H.Free;
end;

procedure TQuantumRegister.ControlledNot(AControlQubit, ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) or
     (AControlQubit >= FNumQubits) then Exit;

  { create the Controlled Not (CNOT) gate and apply it }
  H := TQuantumLogicGate.ControlledNot;
  ApplyTwoQubitGate(H, [AControlQubit, ATargetQubit]);
  //ApplyControlledKQubitGate(H, [AControlQubit], [ATargetQubit]);
  H.Free;
end;

procedure TQuantumRegister.ControlledZ(AControlQubit, ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) or
     (AControlQubit >= FNumQubits) then Exit;

  { create the Controlled Pauli-Z (CZ) gate and apply it }
  H := TQuantumLogicGate.ControlledZ;
  ApplyTwoQubitGate(H, [AControlQubit, ATargetQubit]);
  H.Free;
end;

procedure TQuantumRegister.Toffoli(AControlQubit1, AControlQubit2,
  ATargetQubit: Byte);
var
  H: TQuantumLogicGate;
begin
  { check the initial conditions sanity }
  if (FNumStates = 0) or
     (ATargetQubit >= FNumQubits) or
     (AControlQubit1 >= FNumQubits) or
     (AControlQubit2 >= FNumQubits) then Exit;

  { create the CCX ApplyKQubitGate and apply it }
  H := TQuantumLogicGate.Toffoli;
  ApplyKQubitGate(H, [AControlQubit1, AControlQubit2, ATargetQubit]);
  H.Free;
end;

procedure TQuantumRegister.Gate(AGate: TQuantumLogicGate; ATargetQubits: TByteDynArray);
var
  QbLen: Byte;
begin
  QbLen := Length(ATargetQubits);

  { check the sanity of the initial conditions }
  if QbLen = 0 then
  begin
    FError := errNoTargetQubits;
    Exit;
  end;

  if QbLen > kqsMaxSimulatedQubits then
  begin
    FError := errTaskTooLarge;
    Exit;
  end;

  { call one of the optimized procedures according to the number of target qubits }
  case QbLen of
    1: ApplyOneQubitGate(AGate, ATargetQubits[0]);
    2: ApplyTwoQubitGate(AGate, ATargetQubits);
    else ApplyKQubitGate(AGate, ATargetQubits);
  end;
end;

begin
  //Randomize;
  {
  PCG_SetSeq_128_SRandom_R(PCGRandom, PCG_128BIT_CONSTANT(0, UInt64(Now)),
    PCG_128BIT_CONSTANT(0, UInt64(Now)));
  }
  PCG_SetSeq_128_SRandom_R(PCGRandom, PCG_128BIT_CONSTANT(0, RDSEED),
    PCG_128BIT_CONSTANT(0, RDSEED));
end.

