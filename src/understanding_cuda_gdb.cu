//Compile : nvcc -g -G -arch=sm_70 src/understanding_cuda_gdb.cu
#include <string.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <mma.h>
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


__global__ void print_hello()
{
    printf("Hello from : blockIdx = [%i %i], threadIdx = [%i %i]\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char *argv[])
{
    dim3 blockD(4,4);     // threads
    dim3 gridD(2,2);      // blocks
    
    print_hello <<<gridD,blockD>>> ();
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    print_hello <<<gridD,blockD>>> ();
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    return 0;
}
