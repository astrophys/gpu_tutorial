#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>
#include <stdint.h>

int main(void){
    int N=4;
    int N2=N*N;
    printf("half         = %i (sizeof = %i)\n"
           "unsigned_int = %i (sizeof = %i)\n"
           "long         = %i (sizeof = %i)\n"
           "short        = %i (sizeof = %i)\n",
           (half)N2, sizeof((half)N2), 
           (unsigned int)N2, sizeof((unsigned int)N2),
           (long)N2, sizeof((long)N2),
           (short)N2, sizeof((short)N2));
    return 0;
}
