/*
Author : Ali Snedden
Date   : 3/25/19
Purpose:
    I recently profiled Gadgetron. It is spending 172s in cudaGetDeviceProperties(!) by 
    calling it 357055 times.  Let's see if I can replicate that.
Debug  : 
Notes  : 
    1. How to Run:
        module load cuda/9.0 
        nvcc --gpu-architecture=compute_70 matrix_multiply.cu    
       How to debug:
        nvcc -g -G --gpu-architecture=compute_70 matrix_multiply_cache_opt_tensor.cu  ### -G generate debug infor for device code
    2. Recall that we can't print to stderr from gpu thread
    3. To debug cuda-gdb ./a.out
    4. Biggest error is in forgetting to allocate memory between the device and host,
       e.g. cudaMallocManaged()
    5. IO is very expensive.  Appears to get flushed on cudaDeviceSynchronize()
    6. Using matrix_multiply<<<1,1>>> == 431s, while cpu version 14s. Clearly there
        is substantial overhead when using running single gpu thread.
    7. If using more than maxThreadsPerBlock, it fails to compute and doesn't emit an 
        error.
        --> After each kernel call, do gpuErrchk( cudaPeekAtLastError() );

Good Weblinks:
    1. Unified Memory : https://devblogs.nvidia.com/unified-memory-cuda-beginners/
    2. Tensor Core example : https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
    3. http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/docs/NVIDIA_CUDA_C_ProgrammingGuide_3.1.pdf
    4. 'Proper' error handling : https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    5. Cannot kill errant threads and cleanly end computation in CUDA
       --> See : https://stackoverflow.com/questions/52116815/is-there-a-way-terminate-host-and-device-program-execution-if-a-cuda-thread-enco
    6. Warp Matrix Functions : https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
    7. Explaination of grid*, block* : https://stackoverflow.com/a/16619633/4021436
    8. Grid Stride Loop : https://devblogs.nvidia.com/even-easier-introduction-cuda/

Future :
    1. Try managing memory directly on Host and Device.

*/
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>

// This is for tensor cores...
#include <mma.h>
using namespace nvcuda; 
using namespace std; 


// This is C++ code - from stackoverflow : https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
/********************************************************
    ARGS:
        cudaError_t code
        const char* file : 
        int line :
    DESCRIPTION:
        Uses macro and inline function b/c it is important to preserve the
        file and line number in the error printing.
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s : %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}




/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
void exit_with_error(char * message){
    fprintf(stderr, "%s", message);
    fflush(stderr);
    exit(1);
}



/********************************************************
    ARGS:
        int argc        : 
        char *argv[]    : 
    DESCRIPTION:
        Can run as 
            ./a.out 
            ./a.out ouputfile
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
int main(int argc, char *argv[])
{
    // Declare variables
    int nDev = 0;      //Number of devices
    int bigNum = 357055;
    time_t start = time(NULL);
    // Print device statistics.
    cudaGetDeviceCount(&nDev);
    for(int j=0; j<bigNum; j++){
        for(int i=0; i<nDev; i++){
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            if(j==0){
                printf("Device : %i, Card : %s\n",i,prop.name);
                printf("warpSize : %d\n", prop.warpSize);
                printf("multiProcessorCount : %d\n", prop.multiProcessorCount);
                printf("maxThreadsPerMultiProcessor : %i\n", prop.maxThreadsPerMultiProcessor);
                printf("maxThreadsPerBlock : %i\n", prop.maxThreadsPerBlock);
                printf("maxGridSize : [%i %i %i]\n", prop.maxGridSize[0],
                        prop.maxGridSize[1], prop.maxGridSize[2]);
            }
        }
        if(j%50000 == 0){
            printf("...%.1f %% done \n",float(j)/bigNum * 100);
        }
    }
    printf("Run time : %.3f s\n", difftime(time(NULL), start));

    return 0;
}

