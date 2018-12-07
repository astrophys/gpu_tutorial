/******************************************************************
Author : Ali Snedden
License: MIT
Date   : 12/6/18
Purpose:

Notes:
1. Nvidia sucks. They should really provide a way of converting 
   ints to halfs on the host. So, in order to print out an array of 
   'halfs', you must send it to the device. This is a real PITA.
   

*******************************************************************/
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>
#include <cuda_fp16.hpp>
#include <stdint.h>
using namespace nvcuda; 
using namespace std; 

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s : %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



/**********************************
ARGS:
RETURN:
DESCRIPTION:
    Map 2D indices to 1D index
DEBUG:
    1. read_numpy_matrix() uses this function extensively.
       Directly compared output from read_numpy_matrix() with input
       and was IDENTICAL. This could not work if map_idx() didn't 
       function correctly.
FUTURE:
    1. Add error checking if not too expensive
***********************************/
int map_idx(int i, int j, int Ny){
    return (Ny * i + j);
}
// Visible to device
__device__ int d_map_idx(int i, int j, int Ny){
    return (Ny * i + j);
}

/**********************************
ARGS:
    int * A : flattened 2D array
    int M   : number of Rows
    int N   : number of Cols
RETURN:
DESCRIPTION:
    Print 2D matrix. Must do it to on device b/c the halfs must
    be converted to ints __and__ that can __only__ be done on 
    the device. It is ridiculous, but I'm only using __one__ 
    thread to print the matrix.
DEBUG:
FUTURE:
***********************************/
__global__ void print_matrix(half * A, int M, int N){
    int i = 0;
    int j = 0;
    int rIdx = blockIdx.x * blockDim.x + threadIdx.x;     //Row    index
    int cIdx = blockIdx.y * blockDim.y + threadIdx.y;     //Column index
    
    if(rIdx == 0 && cIdx == 0){
        for(i=0; i<M; i++){
            for(j=0; j<N; j++){
                printf("%*i", 3, __half2int_rd(A[d_map_idx(i,j,N)]));
            }
            printf("\n");
        }
    }
}



/**********************************
ARGS:
    int * A : flattened 2D array - Input array to convert
    half * B: flattened 2D array - Result
    int M   : number of Rows
    int N   : number of Cols
RETURN:
DESCRIPTION:
    Print 2D matrix
DEBUG:
FUTURE:
***********************************/
__global__ void convert_int_to_half(int * A, half * B, int M, int N){
    int rIdx = blockIdx.x * blockDim.x + threadIdx.x;     //Row    index
    int cIdx = blockIdx.y * blockDim.y + threadIdx.y;     //Column index
    int rStride = blockDim.x * gridDim.x;  //
    int cStride = blockDim.y * gridDim.y;  //

    for(int i=rIdx; i<M; i+=rStride){
        for(int j=cIdx; j<M; j+=cStride){
            B[d_map_idx(i,j,N)] = __int2half_rd(A[d_map_idx(i,j,N)]);
        }
    }
}



/**********************************
ARGS:
    int * A : flattened 2D array - Input array to convert
    half * B: flattened 2D array - Result
    int M   : number of Rows
    int N   : number of Cols
RETURN:
DESCRIPTION:
    Print 2D matrix
DEBUG:
FUTURE:
***********************************/
__global__ void some_func(half * A){
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index of current thread in block
    int stride   = blockDim.x * gridDim.x;                // Number of threads in the block
    //printf("%i : %i : %i \n", startIdx, stride, threadIdx.x);

    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x ==1 && threadIdx.y == 1){
        printf("****************************\n\tblockDim.x = %i\n\tblockDim.y = %i\n\tgridDim.x = %i\n\tgridDim.y = %i\n\tblockIdx.x = %i\n\tblockIdx.y = %i\n\tthreadIdx.x = %i\n\tthreadIdx.y = %i\n",
               blockDim.x, blockDim.y, gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y);
    }
}





int main(void){
    int i = 0;
    int N=4;
    int N2=N*N;
    int  * a = NULL;
    half * b = NULL;
    dim3 blockD(4,4,1);     // Must be used to have multidimensional grid-stride loops
    dim3 gridD(2,2,1);        // Must be used to have multidimensional grid-stride loops
    gpuErrChk(cudaMallocManaged(&a, N2 * sizeof(int)));
    gpuErrChk(cudaMallocManaged(&b, N2 * sizeof(half)));
    //Initialize
    for(i=0; i<N2; i++){
        a[i] = i;
    }

    convert_int_to_half<<<gridD,blockD>>> (a,b,N,N);
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    print_matrix<<<gridD,blockD>>> (b,N,N);
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    //some_func<<<gridD,blockD>>>(a);
    //some_func<<<2,3>>>();
    //gpuErrChk(cudaPeekAtLastError());
    //gpuErrChk(cudaDeviceSynchronize());


    return 0;
}
