unit KQS.Complex;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Complex
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
{$DEFINE KQS_DOUBLE_PRECISION}

interface

uses
  Classes, SysUtils, Math;

type
  {$IFDEF KQS_DOUBLE_PRECISION}
  Real = Double;
  {$ELSE}
  Real = Single;
  {$ENDIF}

const
  kqsPi: Real = 3.1415926535897932384626433832795;
  kqsEuler: Real = 2.718281828459045235360287471352;
  kqs1OverSqrt2: Real = 0.7071067811865476;

type
  TRealDynArray = array of Real;

  Complex = record
    Re, Im: Real;
  end;

  TComplexArray = array[0..8191] of Complex;
  PComplexArray = ^TComplexArray;

  function Cplx(ARe, AIm: Real): Complex;

  operator explicit(A: Complex) S: string;
  operator explicit(A: Complex) R: Real;
  operator explicit(A: TRealDynArray) C: Complex;

  operator :=(R: Real) C: Complex;
  operator :=(C: Complex) R: Real;

  operator +(A, B: Complex) C: Complex;
  operator -(A, B: Complex) C: Complex;
  operator *(A, B: Complex) C: Complex;
  operator /(A, B: Complex) C: Complex;

  function CAbs(A: Complex): Real; inline;
  function CAbsSquared(A: Complex): Real; inline;
  function CMag(A: Complex): Real; inline;
  function CScale(A: Complex; F: Real): Complex; inline;
  function Conj(A: Complex): Complex; inline;
  function CAdd(A, B: Complex): Complex; inline;
  function CSub(A, B: Complex): Complex; inline;
  function CMul(A, B: Complex): Complex; inline;
  function CDiv(A, B: Complex): Complex; inline;

  function CEulerPower(A: Complex): Complex; inline;

implementation

//
//       UTILITY FUNCTIONS
//

{ @param(ARe is the real part of the newly assigned complex number)
  @param(AIm)
  @returns(The initialized complex number)
}
function Cplx(ARe, AIm: Real): Complex;
begin
  Result.Re := ARe;
  Result.Im := AIm;
end;

//
//       OPERATORS
//       (functions performing mathematical operations
//       on complex numbers)
//

operator explicit(A: Complex) S: string;
begin
  if A.Im >= 0.0 then
    S := '(' + FloatToStrF(A.Re, ffFixed, 1, 2) + ' + ' +
      FloatToStrF(A.Im, ffFixed, 1, 2) + 'i)'
  else
    S := '(' + FloatToStrF(A.Re, ffFixed, 1, 2) + ' - ' +
      FloatToStrF(-A.Im, ffFixed, 1, 2) + 'i)';
end;

operator explicit(A: Complex) R: Real;
begin
  R := A.Re;
end;

{
  This explicit typecast operator enables initialization of complex
  numbers via assignment of explicitly cast array of two real numbers
  as Complex, i.e., like this:
  C := Complex([2, 5]);

  @param(A is an array of real numbers, passed as literal, where the
  first item A[0] plays the role of the real component, and the second
  item A[1] plays the role the the imaginary component of the complex
  number)
  @returns(The initialized complex number)
}
operator explicit(A: TRealDynArray) C: Complex;
begin
  C.Re := A[0];
  C.Im := A[1];
end;

operator :=(R: Real) C: Complex;
begin
  C.Re := R;
  C.Im := 0.0;
end;

operator :=(C: Complex) R: Real;
begin
  R := C.Re;
end;

operator +(A, B: Complex) C: Complex;
begin
  C.Re := A.Re + B.Re;
  C.Im := A.Im + B.Im;
end;

operator -(A, B: Complex) C: Complex;
begin
  C.Re := A.Re - B.Re;
  C.Im := A.Im - B.Im;
end;

operator *(A, B: Complex) C: Complex;
begin
  C.Re := A.Re * B.Re - A.Im * B.Im;
  C.Im := A.Re * B.Im + A.Im * B.Re;
end;

operator /(A, B: Complex) C: Complex;
var
  D: Real;
begin
  D := B.Re * B.Re + B.Im * B.Im;

  C.Re := (A.Re * B.Re + A.Im * B.Im) / D;
  C.Im := (A.Im * B.Re - A.Re * B.Im) / D;
end;

//
//       ARITHMETIC FUNCTIONS
//       (for the case an operation needs to be
//       called as a function)
//

function CAbs(A: Complex): Real; inline;
begin
  Result := Sqrt(A.Re * A.Re + A.Im * A.Im);
end;

function CAbsSquared(A: Complex): Real; inline;
begin
  Result := A.Re * A.Re + A.Im * A.Im;
end;

function CMag(A: Complex): Real; inline;
begin
  Result := Sqrt(A.Re * A.Re + A.Im * A.Im);
end;

function CScale(A: Complex; F: Real): Complex; inline;
begin
  Result.Re := F * A.Re;
  Result.Im := F * A.Im;
end;

function Conj(A: Complex): Complex; inline;
begin
  Result.Re := A.Re;
  Result.Im := -A.Im;
end;

function CAdd(A, B: Complex): Complex; inline;
begin
  Result := A + B;
end;

function CSub(A, B: Complex): Complex; inline;
begin
  Result := A - B;
end;

function CMul(A, B: Complex): Complex; inline;
begin
  Result := A * B;
end;

function CDiv(A, B: Complex): Complex; inline;
begin
  Result := A / B;
end;

function CEulerPower(A: Complex): Complex; inline;
begin
  Result.Re := Power(kqsEuler, A.Re) * Cos(A.Im);
  Result.Im := Power(kqsEuler, A.Re) * Sin(A.Im);
end;

begin
  kqs1OverSqrt2 := Real(1.0 / Sqrt(2.0));
end.

