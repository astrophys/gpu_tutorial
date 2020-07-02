/***********************************************************************************
Author : Ali Snedden
Date   : 3/25/19
Purpose:
    I recently profiled Gadgetron. It is spending 172s in cudaGetDeviceProperties(!) by 
    calling it 357055 times.  Let's see if I can replicate that.
Debug  : 
Notes  : 
    1. How to Run:
            module load cuda/9.0 
            nvcc --gpu-architecture=compute_35 k40_test.cu
       How to debug:
            nvcc -g -G -Wall --gpu-architecture=compute_35 src/k40_test.cu
            cuda-gdb ./a.out
    2. Recall that we can't print to stderr from gpu thread
    3. IO is very expensive.  Appears to get flushed on cudaDeviceSynchronize()
    4. Using matrix_multiply<<<1,1>>> == 431s, while cpu version 14s. Clearly there
        is substantial overhead when using running single gpu thread.
    5. If using more than maxThreadsPerBlock, it fails to compute and doesn't emit an 
        error.
        --> After each kernel call, do gpuErrchk( cudaPeekAtLastError() );


Resources :
    1. http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/docs/NVIDIA_CUDA_C_ProgrammingGuide_3.1.pdf
    2. 'Proper' error handling 
       --> See https://stackoverflow.com/q/14038589/4021436
    3. Cannot kill errant threads and cleanly end computation in CUDA
       --> See : https://stackoverflow.com/q/52116815/4021436
    4. Warp Matrix Functions : 
       --> See : https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma
    5. Explaination of grid*, block* :
       --> See : https://stackoverflow.com/a/16619633/4021436
    6. Grid Stride Loop :
       --> See : https://devblogs.nvidia.com/even-easier-introduction-cuda/
    7. Query GPU device : 
       --> See : https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
    8. How to inspect device memory : 
       --> https://stackoverflow.com/a/37888060/4021436

Future :
    
***********************************************************************************/

//#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>


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
        1. This is C++ code - from stackoverflow : 
           --> See : https://stackoverflow.com/q/14038589/4021436
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
void write_1D_array(float * array1D, int Nx, int Ny, FILE * f){
    int i = 0;
    int j = 0;
    int idx = 0;
    for(i=0; i<Nx; i++){
        for(j=0; j<Ny; j++){
            idx = map_idx(i,j,Ny);
            fprintf(f, "%*.1f ", 5, array1D[idx]);
        }
        fprintf(f, "\n");
    }
}


/******************************************************
ARGS:
    path = path to file to read
    dim  = dimension of returned matrix, expected to be len = 2
RETURN:
DESCRIPTION:
    Map 2D indices to 1D index
DEBUG:
    1. Printed out read in matrix. used 'diff' to compare 
       with original. Was IDENTICAL
       --> This function WORKS!
FUTURE:
    1. Add error checking if not too expensive
*******************************************************/
float * read_numpy_matrix(char* path, int * dim){
    char * line= NULL;
    char * entireFile = NULL;
    char * pch = NULL;  // Used for parsing strings w strtok
    char errStr[500];
    int fileSize = -1;
    int nline = 0;
    int maxchar = 0;    // Maximum number of characters in a lines
    int nchar = 0;      // Number of characters in line
    int ncols = -1;     // Ncolumns in each row.. should be the same for each row
    int ncolsThisRow = 0;   
    int i = 0;
    int j = 0;
    int n = 0;          // Index to loop thru _all_ file chars
    float * matrix = NULL;
    FILE * f = fopen(path, "r");

    printf("\treading : %s\n", path);
    fflush(stdout);

    //Error check
    if(f == NULL){
        sprintf(errStr, "ERROR!!! %s cannot be opened", path);
        exit_with_error(errStr);
    }
    //Get file size
    fseek(f, 0, SEEK_END);
    fileSize = ftell(f);    // Total num chars in file
    rewind(f);

    //Read entire file
    entireFile = (char* )malloc(sizeof(char) * fileSize);
    fread(entireFile, sizeof(char), fileSize, f);
    rewind(f);

    //Find number of lines and maxchar per line...
    for(n=0; n<fileSize; n++){
        if(entireFile[n] == ' '){
            ncolsThisRow++;
        }
        
        if(entireFile[n] == '\n'){
            maxchar = nchar > maxchar ? nchar : maxchar;

            //Must set at first
            if(nline == 0){
                ncols = ncolsThisRow;
            //Oops, rows aren't the same size.
            }else if(ncols != ncolsThisRow){
                sprintf(errStr, "ERROR!!! nchar %i != ncolsThisRow %i\n", nchar, ncolsThisRow);
                exit_with_error(errStr);
            }
            ncolsThisRow=0;
            nchar = 0;
            nline++;
        }
        nchar++;
    }
    maxchar = maxchar + 1; //+1 for null terminator?
    printf("dim = [nline, ncols] =  [%i, %i],  maxchar = %i \n", nline, ncols, maxchar);
    fflush(stdout);
    
    // Done with busy work - now allocate memory, read in array
    matrix = (float *)malloc(nline * maxchar * sizeof(float));
    line   = (char *)malloc(sizeof(char) * maxchar);
    i = 0;
    while(feof(f) == 0){
        if(fgets(line, maxchar, f)){
            //printf("\tEnd of File Reached\n\n");
            //sprintf(errStr, "ERROR!!! in reading 'line'\n");
            //exit_with_error(errStr);
        }
        // Parse line in file
        pch = strtok(line," ");
        j = 0;
        while(pch != NULL){
            matrix[map_idx(i,j,ncols)] = (float)atof(pch);
            pch = strtok(NULL, " ");
            j++;
        }
        i++;
    }

    /* Debug 
    for(i=0; i<nline; i++){
        for(j=0; j<ncols; j++){
            printf("%.1f ", matrix[map_idx(i,j,ncols)]);
        }
        printf("\n");
    }*/
    
    free(line);
    free(entireFile);
    fclose(f);
    dim[0] = nline;
    dim[1] = ncols;
    return matrix;
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
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x; // Idx of cur thread in block
    int stride   = blockDim.x * gridDim.x;                // Num of threads in block
    int ai = 0;         // Index iterating over rows in A
    int bj = 0;         // Index iterating over columns in B
    float sum = 0;
    printf("%i %i : [%i %i] %i %i\n", startIdx, stride, threadIdx.x, blockIdx.x,
            blockDim.x, gridDim.x);
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
                // EXPENSIVE!! increases runtime 100x
                //printf("\t[%i, %i] x [%i, %i]\n", ai, j, j, bj);

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
    int nDev = 0;

    // Print CUDA devices
    cudaGetDeviceCount(&nDev);
    for (int i = 0; i < nDev; i++) {
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

    // Declare variables
    int i = 0;                  // An index
    int nDim = 2;
    char path[100];
    char errStr[200];
    // Host 
    int * h_dimA = (int *)malloc(sizeof(int) * nDim);        // 
    int * h_dimB = (int *)malloc(sizeof(int) * nDim);        //
    int * h_dimAB= (int *)malloc(sizeof(int) * nDim);        //
    float * h_A = NULL;         // input matrix A.txt , 2D mapped to 1D
    float * h_B = NULL;         // input matrix B.txt , 2D mapped to 1D
    float * h_AB = NULL;        // result from A*B, recorded in AB.txt
    // Device
    int * d_dimA = NULL;        // 
    int * d_dimB = NULL;        // 
    int * d_dimAB= NULL;        // 
    gpuErrChk(cudaMalloc(&d_dimA,  sizeof(int) * nDim));
    gpuErrChk(cudaMalloc(&d_dimB,  sizeof(int) * nDim));
    gpuErrChk(cudaMalloc(&d_dimAB, sizeof(int) * nDim));
    float * d_A = NULL;         // input matrix A.txt , 2D mapped to 1D
    float * d_B = NULL;         // input matrix B.txt , 2D mapped to 1D
    float * d_AB = NULL;        // result from A*B, recorded in AB.txt

    float *answer = NULL;       // result using CUDA of A*B
    FILE * fout = NULL;         // result from answer, written to file

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
        printf("maxGridSize : %i\n", prop.maxGridSize);
        printf("Running matrix_multiply.cu...\n");
    }


    // Read in data to host
    //sprintf(path, "data/very_small/A.txt");
    sprintf(path, "data/small/A.txt");
    h_A = read_numpy_matrix(path, h_dimA);
    //sprintf(path, "data/very_small/B.txt");
    sprintf(path, "data/small/B.txt");
    h_B = read_numpy_matrix(path, h_dimB);
    //sprintf(path, "data/very_small/AB.txt");
    sprintf(path, "data/small/AB.txt");
    answer = read_numpy_matrix(path, h_dimAB);
    h_dimAB[0] = h_dimA[0];
    h_dimAB[1] = h_dimB[1];
    h_AB = (float *)malloc(h_dimAB[0] * h_dimAB[1] * sizeof(float));

    // Allocate device objects
    // Transfer data from host to device
    gpuErrChk(cudaMemcpy(d_dimA, h_dimA, nDim * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_dimB, h_dimB, nDim * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMalloc(&d_A,  h_dimA[0] * h_dimA[1] * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_B,  h_dimB[0] * h_dimB[1] * sizeof(float)));
    gpuErrChk(cudaMalloc(&d_AB, h_dimAB[0] * h_dimAB[1] * sizeof(float)));

    gpuErrChk(cudaMemcpy(&d_dimAB[0], &h_dimAB[0], sizeof(int),cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(&d_dimAB[1], &h_dimAB[1], sizeof(int),cudaMemcpyHostToDevice));
    
    gpuErrChk(cudaMemcpy(d_A, h_A, h_dimA[0] * h_dimA[1] * sizeof(float),
                         cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_B, h_B, h_dimB[0] * h_dimB[1] * sizeof(float),
                         cudaMemcpyHostToDevice));

    time_t start = time(NULL);
    //matrix_multiply<<<1,1>>> (d_A, d_B, d_dimA, d_dimB, d_AB, d_dimAB);  
    //matrix_multiply<<<1,32>>> (d_A, d_B, d_dimA, d_dimB, d_AB, d_dimAB);  
    matrix_multiply<<<32,32>>> (d_A, d_B, d_dimA, d_dimB, d_AB, d_dimAB);  
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());
    cudaDeviceSynchronize();     // Need to block here to prevent further CPU execution
    // Copy back to Host
    gpuErrChk(cudaMemcpy(h_AB, d_AB, h_dimAB[0] * h_dimAB[1] * sizeof(float),
                         cudaMemcpyDeviceToHost))

    printf("Run time : %.3f s\n", difftime(time(NULL), start));
    if(argc == 1){
        fout = fopen("output/AB_result.txt", "w+");
    }else if(argc == 2){
        fout = fopen(argv[1], "w+");
    }else{
        sprintf(errStr, "ERROR!!! Incorrect number of arguments");
        exit_with_error(errStr);
    }

    write_1D_array(h_AB, h_dimAB[0], h_dimAB[1], fout);
    fclose(fout);


    // Release memory
    gpuErrChk(cudaFree(d_A));
    gpuErrChk(cudaFree(d_B));
    gpuErrChk(cudaFree(d_AB));
    gpuErrChk(cudaFree(d_dimA));
    gpuErrChk(cudaFree(d_dimB));
    gpuErrChk(cudaFree(d_dimAB));
    free(h_A);
    free(h_B);
    free(h_AB);
    free(h_dimA);
    free(h_dimB);
    free(h_dimAB);
    


    return 0;
}
