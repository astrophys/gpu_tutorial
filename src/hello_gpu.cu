/*
Author : Ali Snedden
Date   : 8/21/18
Purpose:
    This is a program that multiplies two matrices.
Debug  : 
Notes  : 
    1. How to Run:
        module load cuda11.8/toolkit/11.8.0
        make
    2. http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/docs/NVIDIA_CUDA_C_ProgrammingGuide_3.1.pdf
    3. Unified Memory : https://devblogs.nvidia.com/unified-memory-cuda-beginners/
    4. Cannot kill errant threads and cleanly end computation in CUDA
       --> See : https://stackoverflow.com/questions/52116815/is-there-a-way-terminate-host-and-device-program-execution-if-a-cuda-thread-enco
    5. 'Proper' error handling : https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
    6. Recall that we can't print to stderr from gpu thread
    7. To debug cuda-gdb ./a.out
    8. Biggest error is in forgetting to allocate memory between the device and host,
       e.g. cudaMallocManaged()
    9. IO is very expensive.  Appears to get flushed on cudaDeviceSynchronize()
    10. Using matrix_multiply<<<1,1>>> == 431s, while cpu version 14s. Clearly there
        is substantial overhead when using running single gpu thread.
    11. If using more than maxThreadsPerBlock, it fails to compute and doesn't emit an 
        error.
        --> After each kernel call, do gpuErrchk( cudaPeekAtLastError() );

Good Weblinks:
    1. Unified Memory : https://devblogs.nvidia.com/unified-memory-cuda-beginners/

Future :
    1. Try managing memory directly on Host and Device.

*/
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>

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
__device__ int d_map_idx(int i, int j, int Ny){
    return (Ny * i + j);
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


/**********************************
ARGS:
    array1D : 'flattened' 2D array as 1D
    N       : length of array
RETURN:
    N/A
DESCRIPTION:
    Prints 1D array and 3D coords
DEBUG:
    1. spot checked, it works
FUTURE:
***********************************/
void print_1D_array(float * array1D, int Nx, int Ny){
    int i = 0;
    int j = 0;
    int idx = 0;
    for(i=0; i<Nx; i++){
        for(j=0; j<Ny; j++){
            idx = map_idx(i,j,Ny);
            printf("%*.1f ", 5, array1D[idx]);
        }
        printf("\n");
    }
}

/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
        1. Use 'flattened' 2D array
    FUTURE:
*******************************************************/
void initialize_matrix(float *A, int * dim, float value){
    for(int i=0; i<dim[0]; i++){
        for(int j=0; j<dim[1]; j++){
            //A[i*dim[0]+j] = value;
            A[map_idx(i,j,dim[1])] = value;
        }       
    }

}


/********************************************************
    ARGS:
        A : 'flattened' 2d matrix
        B : 'flattened' 2d matrix
        dimA : gives x & y dims
        dimB : gives x & y dims
        dimAB: pointer modified to return size of new matrix

    DESCRIPTION:
        Multiply A*B : Check dims. Expect only 2 dimensions
        for dimA and dimB.
    RETURN:
    DEBUG:
        1. created code, matrix_generator.py, that multiplies two matrices and
           saves the input and output to a file. I read in data/A.txt, data/B.txt
           and used this function to multiply the matrices. Printed the output and 
           compared to data/AB.txt. It was IDENTICAL. 
           --> This function works!
    NOTES: 
    FUTURE:
*******************************************************/
float * cpu_matrix_multiply(float * A, float * B, int * dimA, int * dimB, int * dimAB)
{
    int j = 0;          // Iterate over elements, do dot product
    int ai = 0;         // Index iterating over rows in A
    int bj = 0;         // Index iterating over columns in B
    float sum = 0;
    char errStr[500];
    float * result = (float *)malloc(sizeof(float) * dimA[0] * dimB[1]);

    // Error Check
    if(dimA[1] != dimB[0]){
        sprintf(errStr, "ERROR!! dimension mismatch, %i != %i", dimA[1], dimB[0]);
        exit_with_error(errStr);
    }

    for(ai=0; ai<dimA[0]; ai++){
        for(bj=0; bj<dimB[1]; bj++){
            sum = 0;
            for(j=0; j<dimA[1]; j++){
                //printf("%.0f * %0.f\n", A[map_idx(ai, j, dimA[1])],
                //        B[map_idx(j, bj, dimB[1])]);

                sum += A[map_idx(ai, j, dimA[1])] * B[map_idx(j, bj, dimB[1])];
                result[map_idx(ai,bj,dimB[1])] = sum;
            }
            //printf("\n");
        }
    }
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    return result;
}


/********************************************************
    ARGS:
        A : 'flattened' 2d matrix
        B : 'flattened' 2d matrix
        dimA : gives x & y dims
        dimB : gives x & y dims
        dimAB: pointer modified to return size of new matrix

    DESCRIPTION:
        Multiply A*B : Check dims. Expect only 2 dimensions
        for dimA and dimB.
    RETURN:
    DEBUG:
    NOTES: 
        1. blockDim.x  : number of threads in each block
           blockIdx.x  : index of current block
           threadIdx.x : 
        2. Error Check - not possible on device code
    FUTURE:
*******************************************************/
__global__ void matrix_multiply(float * A, float * B, int * dimA, int * dimB,
                                float * AB, int * dimAB)
{
    int j = 0;          // Iterate over elements, do dot product
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x; // Index of current thread in block
    int stride   = blockDim.x * gridDim.x;                // Number of threads in the block
    int ai = 0;         // Index iterating over rows in A
    int bj = 0;         // Index iterating over columns in B
    float sum = 0;
    //printf("%i %i : [%i %i] %i %i\n", startIdx, stride, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
    if(blockIdx.x == 0 && threadIdx.x ==0){
        printf("****************************\n\tblockDim.x = %i\n\tgridDim.x = %i\n",
               blockDim.x, gridDim.x);
    }

    // if(dimA[1] != dimB[0]){
    //     char errStr[] = "ERROR!! dimension mismatch\n";
    //     //sprintf(errStr, "ERROR!! dimension mismatch, %i != %i", dimA[1], dimB[0]);
    //     d_exit_with_error(errStr);
    // }
    
    // Grid-stride loop
    for(ai=startIdx; ai<dimA[0]; ai+=stride){
        //printf("[%i %i] : %i : dimA[0] = %i\n", threadIdx.x, blockIdx.x, ai, dimA[0]);
        for(bj=0; bj<dimB[1]; bj++){
            sum = 0;
            for(j=0; j<dimA[1]; j++){
                //printf("\t[%i, %i] x [%i, %i]\n", ai, j, j, bj);  // EXPENSIVE!! increases runtime 100x

                sum += A[d_map_idx(ai, j, dimA[1])] * B[d_map_idx(j, bj, dimB[1])];
                AB[d_map_idx(ai,bj,dimB[1])] = sum;
            }
            //printf("\n");
        }
    }
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
    int * dimA = NULL; //{2,3};
    int * dimB = NULL; //{3,2};
    int * dimAB = NULL; //{0,0};    // Initialize to some value
    float *A = NULL;
    float *B = NULL;
    float *AB = NULL;
    // Print device statistics.
    cudaGetDeviceCount(&nDev);
    for(int i=0; i<nDev; i++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device : %i, Card : %s\n",i,prop.name);
        printf("warpSize : %d\n", prop.warpSize);
        printf("multiProcessorCount : %d\n", prop.multiProcessorCount);
        printf("maxThreadsPerMultiProcessor : %i\n", prop.maxThreadsPerMultiProcessor);
        printf("maxThreadsPerBlock : %i\n", prop.maxThreadsPerBlock);
        printf("maxGridSize : %i\n", *prop.maxGridSize);
        printf("Running matrix_multiply.cu...\n");
    }


    // This uses CUDA's Unified Memory
    gpuErrChk(cudaMallocManaged(&dimA, 2 * sizeof(float)));
    gpuErrChk(cudaMallocManaged(&dimB, 2 * sizeof(float)));
    gpuErrChk(cudaMallocManaged(&dimAB, 2 * sizeof(float)));
    // Set dimensions
    dimA[0] = 2;
    dimA[1] = 3;
    dimB[0] = 3;
    dimB[1] = 2;
    gpuErrChk(cudaMallocManaged(&A, dimA[0] * dimA[1] * sizeof(float)));
    gpuErrChk(cudaMallocManaged(&B, dimB[0] * dimB[1] * sizeof(float)));
    
    // Initialize
    initialize_matrix(A,dimA,2);
    initialize_matrix(B,dimB,4);
    // Set my own values
    A[map_idx(0,0,dimA[1])] = 1;
    A[map_idx(0,1,dimA[1])] = 2;
    A[map_idx(0,2,dimA[1])] = 3;
    A[map_idx(1,0,dimA[1])] = 4;
    A[map_idx(1,1,dimA[1])] = 5;
    A[map_idx(1,2,dimA[1])] = 6;

    B[map_idx(0,0,dimB[1])] = 7;
    B[map_idx(0,1,dimB[1])] = 10;
    B[map_idx(1,0,dimB[1])] = 8;
    B[map_idx(1,1,dimB[1])] = 11;
    B[map_idx(2,0,dimB[1])] = 9;
    B[map_idx(2,1,dimB[1])] = 12;
    
    AB = cpu_matrix_multiply(A, B, dimA, dimB, dimAB);

    // Print matrices
    printf("CPU Multiplying Trivial Matrices\n");
    printf("A (%i x %i):\n", dimA[0], dimA[1]);
    print_1D_array(A, dimA[0], dimA[1]);
    printf("B (%i x %i):\n", dimB[0], dimB[1]);
    print_1D_array(B, dimB[0], dimB[1]);
    printf("AB (%i x %i):\n", dimAB[0], dimAB[1]);
    print_1D_array(AB, dimAB[0], dimAB[1]);
    

    // Try CUDA version of matrix_multiply
    free(AB);
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    gpuErrChk(cudaMallocManaged(&AB, dimAB[0] * dimAB[1] * sizeof(float)));


    time_t start = time(NULL);
    matrix_multiply<<<1024,32>>> (A, B, dimA, dimB, AB, dimAB);  // Fails b/c maxThreadsPerBlock=1024

    // Print matrices
    gpuErrChk(cudaDeviceSynchronize());
    printf("\n\nGPU Multiplying Trivial Matrices\n");
    printf("A (%i x %i):\n", dimA[0], dimA[1]);
    print_1D_array(A, dimA[0], dimA[1]);
    printf("B (%i x %i):\n", dimB[0], dimB[1]);
    print_1D_array(B, dimB[0], dimB[1]);
    printf("AB (%i x %i):\n", dimAB[0], dimAB[1]);
    print_1D_array(AB, dimAB[0], dimAB[1]);
    


    // Read matrix files
    gpuErrChk(cudaFree(A));
    gpuErrChk(cudaFree(B));

    gpuErrChk( cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    return 0;
}

