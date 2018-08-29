/*
Author : Ali Snedden
Date   : 8/16/18
Purpose: 
    Learn CUDA!
Debug  : 
Notes  : 
    1. Inspired by https://devblogs.nvidia.com/even-easier-introduction-cuda/
    2. To compile : 
       module load nvcc/8.0
       module load gcc-4.9.2
       nvcc tutorial.cu  
       nvprof ./a.out       # Profile it!
Future : 
    1. Try P100 and V100 page swapping examples.  
    
       See : https://devblogs.nvidia.com/unified-memory-cuda-beginners/
*/

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>

/********************************************************
    ARGS:
        int n     : 0->n elements are added element wise
        float * x : array 1
        float * y : array 2
    DESCRIPTION:
        function to add the N first elements of two arrays
    RETURN:
        Returns nothing. Modifies y[] in place
    DEBUG:
    FUTURE:
*******************************************************/
__global__  // Denotes it is a CUDA kernel function
void add(int n, float * x, float * y)
{
    //printf("Memory Address of x (kernel) = %p\n", x);
    //printf("Memory Address of y (kernel) = %p\n", y);

    int i = 0;
    int tidx = threadIdx.x;     // Index of current thread in block
    int stride = blockDim.x;    // Number of threadsin the block
    for(i=tidx; i<n; i += stride){
        y[i] = y[i] + x[i];
    }
}



/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    FUTURE:
*******************************************************/
__global__  // Denotes it is a CUDA kernel function
void print_hello(void)
{
    int a = 6;
    // Number of threadsin the block
    printf("blockidx : %i  blockdim : %i  tidx : %i   %p\n", blockIdx.x, blockDim.x, threadIdx.x);
}



/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
        1. <<< , num_threads_per_block >>> is the 'execution configuration
        2. CUDA GPUs run kernels usin blocks of threads that are multiple of 32 in
           size
    FUTURE:
*******************************************************/
int main(void)
{
    time_t start = time(NULL);
    int N = pow(2,20);      // == 1<<20
    printf("N = %i\n", N);
    //float *x = new float[N];
    //float *y = new float[N];
    float *x = NULL;
    float *y = NULL;
    // Allocate memory that is accessible by both GPU and CPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    printf("Memory Address of x (cpu) = %p\n", x);
    printf("Memory Address of y (cpu) = %p\n", y);
    
    // Get device properties
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        /* Properties 'cudaDevideProp'
         name, totalGlobalMem, sharedMemPerBlock, regsPerBlock, warpSize,
         memPitch, maxThreadsPerBlock, maxThreadsDim = {1024, 1024, 64},
         maxGridSize, clockRate, totalConstMem, major, minor, textureAlignment,
         texturePitchAlignment, deviceOverlap, multiProcessorCount, 
         kernelExecTimeoutEnabled, integrated, canMapHostMemory, computeMode,
         maxTexture1D, maxTexture1DMipmap, maxTexture1DLinear, maxTexture2D,
         maxTexture2DMipmap, maxTexture2DLinear, maxTexture2DGather, maxTexture3D,
         maxTexture3DAlt, maxTextureCubemap, maxTexture1DLayered, maxTexture2DLayered,
         maxTextureCubemapLayered, maxSurface1D, maxSurface2D, maxSurface3D,
         maxSurface1DLayered, maxSurface2DLayered, maxSurfaceCubemap, 
         maxSurfaceCubemapLayered, surfaceAlignment, concurrentKernels, ECCEnabled,
         pciBusID, pciDeviceID, pciDomainID, tccDriver, asyncEngineCount, 
         unifiedAddressing, memoryClockRate, memoryBusWidth, l2CacheSize, 
         maxThreadsPerMultiProcessor, streamPrioritiesSupported, globalL1CacheSupported,
         localL1CacheSupported, sharedMemPerMultiprocessor, regsPerMultiprocessor, 
         managedMemory, isMultiGpuBoard, multiGpuBoardGroupID, hostNativeAtomicSupported,
         singleToDoublePrecisionPerfRatio, pageableMemoryAccess, concurrentManagedAccess */
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }



    // Initialize
    for(int i = 0; i < N; i++){
        x[i] = 1.0;
        y[i] = 2.0;
    }

    // Run
    //add(N, x, y);
    add<<<1,32>>>(N, x, y);      
    print_hello<<<2,6>>>();     
    // Block CPU until GPU is finished. GPU calls are non-blocking
    cudaDeviceSynchronize();
    
    // Error Check
    //cudaError_t errSync  = cudaGetLastError();
    //cudaError_t errAsync = cudaDeviceSynchronize();
    //if (errSync != cudaSuccess) 
    //    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    //if (errAsync != cudaSuccess)
    //    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));


    // Check for error, all entries should == 3.0. 
    // 3.0_f_, to cast as float, not default double.
    float maxErr = 0.0;
    for(int i=0; i<N; i++){
        maxErr = fmax(maxErr, fabs(y[i] - 3.0f));
    }
    printf("Max Error : %f\n", maxErr);

    // Free memory
    // delete [] x;
    // delete [] y;
    cudaFree(x);
    cudaFree(y);

    time_t end = time(NULL);
    printf("Run time : %.3f s\n", difftime(end, start));
    return 0;
}





