unit KQS.Algebra;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Algebra
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
  Classes, SysUtils, KQS.Complex;

type
  { TComplexVector }
  TComplexVector = class(TObject)
  private
    FAllocated: Boolean;
    FSize: Cardinal;
    FComponents: PComplexArray;
  public
    property Size: Cardinal read FSize;
    property Components: PComplexArray read FComponents;

    constructor Create(ASize: Cardinal);
    constructor Copy(AVector: TComplexVector);
    destructor Destroy; override;

    function GetComponent(AIndex: Cardinal): Complex;
    procedure SetComponent(AIndex: Cardinal; AValue: Complex);

    function ToString: string; override;
    procedure Print;
  end;

  { TComplexMatrix }
  TComplexMatrix = class(TObject)
  private
    FAllocated: Boolean;
    FRows, FColumns: Cardinal;
    FElements: TComplexVector;
  public
    property Rows: Cardinal read FRows;
    property Columns: Cardinal read FColumns;
    property Elements: TComplexVector read FElements;

    constructor Create(ARows, AColumns: Cardinal);
    destructor Destroy; override;

    function GetElement(ARow, ACol: Cardinal): Complex; inline;
    procedure SetElement(ARow, ACol: Cardinal; AValue: Complex); inline;

    procedure Print;
  end;

  //
  //     AUX FUNCTION
  //

  function TwoPower(A: Byte): QWord;
  function StrToBin(S: string): Cardinal;
  function BinToStr(A: Cardinal; L: Byte): string;

implementation

{ ______________________________________________________________________________

         TComplexVector
         CLASS IMPLEMENTATION
  ______________________________________________________________________________
}

constructor TComplexVector.Create(ASize: Cardinal);
begin
  inherited Create;

  FAllocated := False;
  FSize := ASize;
  ReturnNilIfGrowHeapFails := False;

  try
    FComponents := GetMem(FSize * SizeOf(Complex));
    FillChar(FComponents^, FSize * SizeOf(Complex), 0);
    FAllocated := True;
  except
    on E: EOutOfMemory do
    begin
      FAllocated := False;
      FSize := 0;
      // TODO: maybe some more sophisticated error management?
    end;
  end;
end;

constructor TComplexVector.Copy(AVector: TComplexVector);
begin
  { check if the object to copy exists }
  if not Assigned(AVector) then Exit;

  { call the original constructor }
  Create(AVector.Size);

  { copy the content of the argument object }
  if FAllocated then
  begin
    Move(AVector.Components^, FComponents^, FSize * SizeOf(Complex));
  end;
end;

destructor TComplexVector.Destroy;
begin
  if FAllocated and (FSize > 0) then
    FreeMem(FComponents, FSize * SizeOf(Complex));

  inherited Destroy;
end;

function TComplexVector.GetComponent(AIndex: Cardinal): Complex;
begin
  if not FAllocated then Exit(Cplx(0, 0));

  if AIndex < FSize then Result := FComponents^[AIndex]
  else Result := Cplx(0, 0);
end;

procedure TComplexVector.SetComponent(AIndex: Cardinal; AValue: Complex);
begin
  if not FAllocated then Exit;

  if AIndex < FSize then FComponents^[AIndex] := AValue;
end;

function TComplexVector.ToString: string;
var
  I: Cardinal;
begin
  if not FAllocated then Exit('(X)');

  Result := '(';

  for I := 0 to FSize - 1 do
  begin
    Result := Result + string(FComponents^[I]);
    if I < FSize - 1 then Result := Result + ' ';
  end;

  Result := Result + ')';
end;

procedure TComplexVector.Print;
begin
  Write(ToString);
end;

{ ______________________________________________________________________________

         TComplexMatrix
         CLASS IMPLEMENTATION
  ______________________________________________________________________________
}

constructor TComplexMatrix.Create(ARows, AColumns: Cardinal);
begin
  inherited Create;

  FAllocated := False;
  FRows := ARows;
  FColumns := AColumns;

  FElements := TComplexVector.Create(FRows * FColumns);
  FAllocated := FElements.FAllocated;

  if FElements.Size = 0 then
  begin
    FRows := 0;
    FColumns := 0;
  end;
end;

destructor TComplexMatrix.Destroy;
begin
  if (FRows > 0) and
     (FColumns > 0) then FElements.Free;

  inherited Destroy;
end;

function TComplexMatrix.GetElement(ARow, ACol: Cardinal): Complex; inline;
begin
  Result := Cplx(0, 0);
  if not FAllocated then Exit;

  if (ARow <= FRows) and (ACol <= FColumns) then
    Result := FElements.Components^[ARow * FColumns + ACol];
end;

procedure TComplexMatrix.SetElement(ARow, ACol: Cardinal; AValue: Complex); inline;
begin
  if not FAllocated then Exit;

  if (ARow <= FRows) and (ACol <= FColumns) then
    FElements.Components^[ARow * FColumns + ACol] := AValue;
end;

procedure TComplexMatrix.Print;
var
  I, J: Cardinal;
begin
  if not FAllocated then
  begin
    WriteLn('|X|');
    Exit;
  end;

  for I := 0 to FRows - 1 do
  begin
    Write('| ');

    for J := 0 to FColumns - 1 do
    begin
      Write(string(FElements.Components^[I * FColumns + J]), ' ');
    end;

    WriteLn('|');
  end;
end;

{ ______________________________________________________________________________

         AUXILIARY FUNCTIONS
  ______________________________________________________________________________
}

{$ASMMODE Intel}
{$IF DEFINED(WIN64)}
  function TwoPower(A: Byte): QWord; register; assembler;
  {$IF DEFINED(CPU64)}
  asm
         MOV   RAX, 1
         SHL   RAX, CL
  end;
  {$ENDIF}
  {$IF DEFINED(CPU32)}
  asm
         MOV   CL, AL
         MOV   EAX, 1
         SHL   EAX, CL
  end;
  {$ENDIF}
{$ELSEIF DEFINED(UNIX) AND DEFINED(CPU64)}
  function PowerOfTwo(A: Byte): QWord; register; assembler;
  asm
         MOV   RCX, RSI
         MOV   RAX, 1
         SHL   RAX, CL
  end;
{$ELSE}
  function PowerOfTwo(A: Byte): QWord;
  begin
    Result := 1 shl A;
  end;
{$ENDIF}

function StrToBin(S: string): Cardinal;
var
  I, P: Cardinal;
begin
  Result := 0;
  P := 0;

  for I := Length(S) downto 1 do
  begin
    if S[I] = '1' then Result := Result or (1 shl P);
    Inc(P);
  end;
end;

function BinToStr(A: Cardinal; L: Byte): string;
var
  I: Cardinal;
begin
  Result := '';
  for I := 0 to L - 1 do Result := Result + '0';


  for I := 0 to L - 1 do
    if ((A shr I) and 1) = 1 then Result[L - I] := '1';
end;

end.

