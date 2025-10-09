program KEQS_Test_External;

{$MODE ObjFPC}
{$H+}

uses ctypes, SysUtils, KQS.Complex, KQS.Algebra, KQS.Circuit, KQS.Simulator, KQS.Random;

function E_Add(a, b: cint): cint; cdecl; external 'Test_External';
function E_Mul(a, b: cint): cint; cdecl; external 'Test_External';
procedure E_Hello(); cdecl; external 'Test_External';

procedure TestExternal;
var
  A: Complex;
begin
  WriteLn;
  WriteLn('*** COMPLEX NUMBER TESTS ***');

  A := Cplx(3, 2);
  WriteLn('A = ', string(A));
  WriteLn('|A| = ', CMag(A):0:6);
  WriteLn('A_re = ', A.Re:0:6);
  WriteLn('A_im = ', A.Im:0:6);

  WriteLn('*** CALLING E_ADD ***');
  WriteLn('E_Add(3, 4) = ', E_Add(3, 4));
  WriteLn('E_Add(5, 6) = ', E_Add(5, 6));

  WriteLn('*** CALLING E_MUL ***');
  WriteLn('E_Mul(3, 4) = ', E_Mul(3, 4));
  WriteLn('E_Mul(5, 6) = ', E_Mul(5, 6));

  E_Hello();
end;

begin
  TestExternal;

  ReadLn;
end.

