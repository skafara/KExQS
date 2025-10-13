unit KQS.Simulator;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Simulator
         Version 1.0

         Copyright (c) Kamil Ekstein, 2025
         All rights reserved.

         Code written by: Kamil Ekstein
         Target compiler: FPC 3.2.2
         Last update on:  05-Sep-2025
         Last update by:  KE
  ______________________________________________________________________________
}

{$MODE ObjFPC}
{$H+}

interface

uses
  Classes, SysUtils, ctypes,
  KQS.Algebra, KQS.Circuit, KQS.Random;

const
  ESimulatorDLL = 'ESimulator';

type
  TCardinalArray = array[0..8191] of Cardinal;
  PCardinalArray = ^TCardinalArray;

  { TQuantumSimulator }

  TQuantumSimulator = class(TObject)
  private
    FError: Byte;
    FRegister: TQuantumRegister;
    FNumStates: Cardinal;
    FStateCounts: PCardinalArray;
  public
    property States: Cardinal read FNumStates;

    constructor Create(ARegister: TQuantumRegister);
    destructor Destroy; override;

    procedure Run(ANumShots: Cardinal);

    function GetStateCount(AState: Cardinal): Cardinal;
  end;

procedure ESimulator_Run(
  AStateCounts: Pointer;
  const AStateAmplitudes: Pointer;
  ANumQubits: cuint;
  ANumStates: cuint;
  ANumShots: cuint
); cdecl; external ESimulatorDLL;

implementation

{ ______________________________________________________________________________

         TQuantumSimulator
         CLASS IMPLEMENTATION
  ______________________________________________________________________________
}

constructor TQuantumSimulator.Create(ARegister: TQuantumRegister);
begin
  inherited Create;

  { pre-initialization }
  FNumStates := 0;
  FRegister := ARegister;

  { check if there is anything to simulate }
  if (FRegister = nil) or
     (FRegister.Qubits = 0) then
  begin
    FError := errEmptyRegister;
    Exit;
  end;

  { check if the simulation is feasible }
  if FRegister.Qubits > kqsMaxSimulatedQubits then
  begin
    FError := errTaskTooLarge;
    Exit;
  end;

  { initialize the simulator }
  FNumStates := TwoPower(FRegister.Qubits);
  GetMem(FStateCounts, FNumStates * SizeOf(Cardinal));
  FError := errOK;
end;

destructor TQuantumSimulator.Destroy;
begin
  { release the allocated memory }
  if FNumStates > 0 then FreeMem(FStateCounts, FNumStates * SizeOf(Cardinal));

  inherited Destroy;
end;

procedure TQuantumSimulator.Run(ANumShots: Cardinal);
{$IFNDEF RUN_EXTERNAL}
var
  N, I: Cardinal;
  B: Byte;
  Qb: Boolean;
{$ENDIF}
begin
  { check if there is anything to simulate }
  if FNumStates = 0 then Exit;
  
  { prepare the simulation }
  FillChar(FStateCounts^, FNumStates * SizeOf(Cardinal), 0);

  {$IFDEF RUN_EXTERNAL}

  ESimulator_Run(FStateCounts, FRegister.Amplitudes_, FRegister.Qubits, FNumStates, ANumShots);

  {$ELSE}

  { the main simulation loop }
  for N := 0 to ANumShots - 1 do
  begin
    I := 0;
    for B := 0 to FRegister.Qubits - 1 do
    begin
      Qb := FRegister.Measure(B, False);
      if Qb then I := I or (1 shl B);
    end;

    { increate the number of hits for the Ith state }
    Inc(FStateCounts^[I]);
  end;

  {$ENDIF}
end;

function TQuantumSimulator.GetStateCount(AState: Cardinal): Cardinal;
begin
  if AState >= FNumStates then Exit(0);
  Result := FStateCounts^[AState];
end;

end.

