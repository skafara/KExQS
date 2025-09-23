unit KQS.Random;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Random
         Version 1.0

         Copyright (c) Kamil Ekstein, 2025
         All rights reserved.

         Code written by: Kamil Ekstein
         Target compiler: FPC 3.2.2
         Last update on:  18-Sep-2025
         Last update by:  KE
  ______________________________________________________________________________
}

{$MODE ObjFPC}
{$H+}

interface

uses
  Classes, SysUtils;

type
  TPCGInt128 = packed record
    High: UInt64;
    Low: UInt64;
  end;

  TPCG64Random = packed record
    State: TPCGInt128;
    Increment: TPCGInt128;
  end;

var
  pcgDefaultMultiplier128: TPCGInt128;
  pcgDefaultIncrement128: TPCGInt128;

  function PCG_128BIT_CONSTANT(AHigh, ALow: UInt64): TPCGInt128; inline;
  procedure PCG_SetSeq_128_SRandom_R(var ARng: TPCG64Random;
    AInitState, AInitSeq: TPCGInt128);
  function PCG_SetSeq_128_XSL_RR_64_Random_R(var ARng: TPCG64Random): UInt64;

  function PCGRandom_UNorm(var ARng: TPCG64Random): Double;
  function RDRAND_UNorm: Double;
  function RDSEED: UInt64;

implementation

function PCG_128BIT_CONSTANT(AHigh, ALow: UInt64): TPCGInt128; inline;
begin
  Result.High := AHigh;
  Result.Low := ALow;
end;

function PCG_ROR64(AValue: UInt64; ARot: Cardinal): UInt64; inline;
begin
  Result := (AValue shr ARot) or (AValue shl ((-ARot) and 63));
end;

procedure PCG_MUL64(X, Y: UInt64; var Z1, Z0: UInt64);
var
  X0, X1, Y0, Y1: UInt64;
  W0, W1, W2, T: UInt64;
begin
  Z0 := X * Y;

  X0 := X and UInt64($FFFFFFFF);
  X1 := X shr 32;
  Y0 := Y and UInt64($FFFFFFFF);
  Y1 := Y shr 32;
  W0 := X0 * Y0;
  T := X1 * Y0 + (W0 shr 32);
  W1 := T and UInt64($FFFFFFFF);
  W2 := T shr 32;
  W1 := W1 + X0 * Y1;
  Z1 := X1 * Y1 + W2 + (W1 shr 32);
end;

function PCG_ADD128(A, B: TPCGInt128): TPCGInt128;
var
  I: Cardinal;
begin
  Result.Low := A.Low + B.Low;
  if Result.Low < B.Low then I := 1 else I := 0;
  Result.High := A.High + B.High + I;
end;

function PCG_MUL128(A, B: TPCGInt128): TPCGInt128;
var
  H1: UInt64;
begin
  H1 := A.High * B.Low + A.Low * B.High;
  PCG_MUL64(A.Low, B.Low, Result.High, Result.Low);
  Result.High := Result.High + H1;
end;

procedure PCG_SetSeq128_Step_R(var ARng: TPCG64Random);
begin
  ARng.State := PCG_ADD128(PCG_MUL128(ARng.State, pcgDefaultMultiplier128), ARng.Increment);
end;

function PCG_Output_XSL_RR_128_64(AState: TPCGInt128): UInt64;
begin
  Result := PCG_ROR64(AState.High xor AState.Low, AState.High shr 58);
end;

procedure PCG_SetSeq_128_SRandom_R(var ARng: TPCG64Random;
  AInitState, AInitSeq: TPCGInt128);
begin
  ARng.State := PCG_128BIT_CONSTANT(0, 0);
  ARng.Increment.High := AInitSeq.High shl 1;
  ARng.Increment.High := ARng.Increment.High or (AInitSeq.Low shr 63);
  ARng.Increment.Low := (AInitSeq.Low shl 1) or 1;
  PCG_SetSeq128_Step_R(ARng);
  ARng.State := PCG_ADD128(ARng.State, AInitState);
  PCG_SetSeq128_Step_R(ARng);
end;

function PCG_SetSeq_128_XSL_RR_64_Random_R(var ARng: TPCG64Random): UInt64;
begin
  PCG_SetSeq128_Step_R(ARng);
  Result := PCG_Output_XSL_RR_128_64(ARng.State);
end;

function PCG_SetSeq_128_XSL_RR_64_BoundedRand_R(var ARng: TPCG64Random;
  ABound: UInt64): UInt64;
var
  Threshold, R: UInt64;
begin
  Threshold := -ABound mod ABound;

  while True do
  begin
    R := PCG_SetSeq_128_XSL_RR_64_Random_R(ARng);
    if R >= Threshold then Exit(R mod ABound);
  end;
end;

function PCGRandom_UNorm(var ARng: TPCG64Random): Double;
var
  R: UInt64;
begin
  R := PCG_SetSeq_128_XSL_RR_64_Random_R(ARng);
  Result := R / High(UInt64);
end;

{$asmmode intel}
function RDRAND_UNorm: Double;
var
  R: UInt64;
begin
  asm
         RDRAND RAX
         //RDSEED   RAX
         MOV    R, RAX
  end;

  Result := R / High(UInt64);
end;

function RDSEED: UInt64; assembler;
asm
         RDSEED RAX
end;

begin
  pcgDefaultMultiplier128 := PCG_128BIT_CONSTANT(2549297995355413924, 4865540595714422341);
  pcgDefaultIncrement128 := PCG_128BIT_CONSTANT(6364136223846793005, 1442695040888963407);
//  pcgStateSetSeq128Initializer :=
end.

