unit KQS.Utils;

{ ______________________________________________________________________________

         KExQS (KEx Quantum Computer Simulator)
         Module KQS.Utils
         Version 1.0
  ______________________________________________________________________________
}

{$MODE ObjFPC}
{$H+}

interface

uses
  Classes, SysUtils;

{ ______________________________________________________________________________
  
         MEMORY MANAGEMENT UTILITIES
  ______________________________________________________________________________
}

{*
  Allocates a block of memory aligned to a specified byte boundary.
  Alignment must be a power of two (8, 16, 32, 64...).
  The function returns a pointer to the aligned memory block.
  The memory must be released using AlignedFree().
*}
function AlignedAlloc(Size, Alignment: NativeUInt): Pointer;

{*
  Frees memory previously allocated by AlignedAlloc().
*}
procedure AlignedFree(P: Pointer);

{*
  Allocates a block of memory aligned for AVX2 (32 bytes).
  Equivalent to calling AlignedAlloc(Size, 32).
*}
function AVX2AlignedAlloc(Size: NativeUInt): Pointer;

{*
  Frees memory previously allocated by AVX2AlignedAlloc().
  Equivalent to AlignedFree(P).
*}
procedure AVX2AlignedFree(P: Pointer);

implementation

{ ______________________________________________________________________________

         MEMORY MANAGEMENT IMPLEMENTATION
  ______________________________________________________________________________
}

function AlignedAlloc(Size, Alignment: NativeUInt): Pointer;
var
  P, PAligned: PByte;
  Offset: NativeUInt;
begin
  if Alignment < SizeOf(Pointer) then
    Alignment := SizeOf(Pointer);

  { Allocate slightly more to ensure space for alignment padding and the back pointer }
  GetMem(P, Size + Alignment - 1 + SizeOf(Pointer));

  { Compute alignment offset }
  Offset := Alignment - ((NativeUInt(P) + SizeOf(Pointer)) and (Alignment - 1));
  PAligned := P + Offset + SizeOf(Pointer);

  { Store the original (unaligned) pointer right before the aligned block }
  PPointer(PAligned - SizeOf(Pointer))^ := P;

  Result := PAligned;
end;

procedure AlignedFree(P: Pointer);
var
  PBase: Pointer;
begin
  if P = nil then Exit;

  { Retrieve original pointer from just before the aligned address }
  PBase := PPointer(NativeUInt(P) - SizeOf(Pointer))^;
  FreeMem(PBase);
end;

function AVX2AlignedAlloc(Size: NativeUInt): Pointer;
begin
  Result := AlignedAlloc(Size, 32);
end;

procedure AVX2AlignedFree(P: Pointer);
begin
  AlignedFree(P);
end;

end.
