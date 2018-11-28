/*
Author : Ali Snedden
Date   : 11/28/18
Purpose:
    This is a program that multiplies two matrices. This combines features from 
    matrix_multiply_omp_cache_optimized.c (e.g. taking the transpose of B) and using
    CUDA.
Debug  : 
Notes  : 
    1. How to Run:
        module load cuda/8.0 
        nvcc matrix_multiply.cu    
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
//__device__ void d_exit_with_error(char * message){
//    fprintf(stderr, "%s", message);
//    fflush(stderr);
//    exit(1);
//}



/**********************************
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
***********************************/
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
    cudaMallocManaged(&matrix, nline * maxchar * sizeof(float));
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
 


/*************************************************************
ARGS:
    float * A   : 2 x 2 matrix, stored as row majored in 1D array
    int   * dim : len(dim) = 2
RETURN:
    -> newM, is a matrix that is column ordered matrix.
    -> dim is unchanged
DESCRIPTION:
    Take Transpose
DEBUG:
    1. Spot checked beginning, middle and end of matrix. It appears
       that I correctly switched from row-majored to column majored
       matrix
FUTURE:
**************************************************************/
float * reorder_row_major_as_col_major(float * B, int * dim){
    int i,j;    // Indices
    float * newM  = NULL;  //(float *)malloc(sizeof(float) * dim[0] * dim[1]); 
    gpuErrChk(cudaMallocManaged(&newM, sizeof(float) * dim[0] * dim[1]));

    //rows
    for(i=0; i<dim[0]; i++){
        for(j=0; j<dim[1]; j++){
            newM[map_idx(j,i,dim[0])]  = B[map_idx(i,j,dim[1])];    // dim[0] or dim[1] for newM?
            //newM[map_idx(i,j,dim[0])]  = B[map_idx(j,i,dim[1])];    // dim[0] or dim[1] for newM?
        }
    }
    printf("Re-ordering matrix B...\n");
    gpuErrChk(cudaFree(B));
    return(newM);
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
    /**** Row of A to multiply ****/
    for(ai=startIdx; ai<dimA[0]; ai+=stride){       
        //printf("[%i %i] : %i : dimA[0] = %i\n", threadIdx.x, blockIdx.x, ai, dimA[0]);

        /**** Column of AB for output and Columns of B ****/
        for(j=0; j<dimB[1]; j++){ 
            sum = 0;
            for(bj=0; bj<dimB[0]; bj++){
                // EXPENSIVE!! increases runtime 100x
                /*printf("\t[%i, %i] x [%i, %i] = %.0f %.0f\n",
                        ai, bj, j, bj, A[d_map_idx(ai, bj, dimA[1])], B[d_map_idx(j, bj, dimB[0])]); */
                sum += A[d_map_idx(ai, bj, dimA[1])] * B[d_map_idx(j, bj, dimB[0])];
            }
            AB[d_map_idx(ai,j,dimB[1])] = sum;
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
    char path[100];
    char errStr[200];
    int nDev = 0;      //Number of devices
    int * dimA = NULL; //{2,3};
    int * dimB = NULL; //{3,2};
    int * dimAB = NULL; //{0,0};    // Initialize to some value
    float *A = NULL;
    float *B = NULL;
    float *AB = NULL;
    float *answer = NULL;
    FILE * fout = NULL;
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
    }


    // This uses CUDA's Unified Memory
    gpuErrChk(cudaMallocManaged(&dimA, 2 * sizeof(float)));
    gpuErrChk(cudaMallocManaged(&dimB, 2 * sizeof(float)));
    gpuErrChk(cudaMallocManaged(&dimAB, 2 * sizeof(float)));
    
    //sprintf(path, "data/very_small/A.txt");
    sprintf(path, "data/large/A.txt");
    A = read_numpy_matrix(path, dimA);
    //sprintf(path, "data/very_small/B.txt");
    sprintf(path, "data/large/B.txt");
    B = read_numpy_matrix(path, dimB);
    time_t start = time(NULL);
    B = reorder_row_major_as_col_major(B, dimB);
    //sprintf(path, "data/AB_small.txt");
    //sprintf(path, "data/large/AB.txt");
    //answer = read_numpy_matrix(path, dimAB);

    // Try CUDA version of matrix_multiply
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    gpuErrChk(cudaMallocManaged(&AB, dimAB[0] * dimAB[1] * sizeof(float)));
    //            <<<gridDim.x (# blocks), blockDim.x (# threads per block) >>>

    matrix_multiply<<<1024,32>>> (A, B, dimA, dimB, AB, dimAB);  // Fails b/c maxThreadsPerBlock=1024
    //matrix_multiply<<<2,3>>> (A, B, dimA, dimB, AB, dimAB);  // Fails b/c maxThreadsPerBlock=1024
    gpuErrChk( cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());

    printf("Run time : %.3f s\n", difftime(time(NULL), start));
    if(argc == 1){
        fout = fopen("output/AB_result.txt", "w+");
    }else if(argc == 2){
        fout = fopen(argv[1], "w+");
    }else{
        sprintf(errStr, "ERROR!!! Incorrect number of arguments");
        exit_with_error(errStr);
    }
    write_1D_array(AB, dimAB[0], dimAB[1], fout);
    fclose(fout);


    return 0;
}

