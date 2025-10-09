#include <stdio.h>


__declspec(dllexport) int E_Add(int a, int b) {
    return a + b;
}

__declspec(dllexport) int E_Mul(int a, int b) {
    return a * b;
}

__declspec(dllexport) void E_Hello() {
    printf("*** Hello World! (C DLL) ***\n");
}
