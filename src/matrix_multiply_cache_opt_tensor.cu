/*
Author : Ali Snedden
Date   : 11/28/18
Purpose:
    This is a program that multiplies two matrices. This combines features from 
    matrix_multiply_omp_cache_optimized.cu (e.g. taking the transpose of B) and using
    CUDA. This code uses the v100 tensor cores.
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
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;



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
    int i  : row Idx
    int j  : col Idx
    int Ny : Number of columns Idx
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
    path = path to file to read
    dim  = dimension of returned matrix, expected to be len = 2
RETURN:
DESCRIPTION:
    Reads matrix generated by matrix_generator.py and 0 pads 
    to make matrix dimensions to be multiples of 16. 
    Function is white space DEPENDANT
DEBUG:
    1. Printed out read in matrix. used 'diff' to compare 
       with original. Was IDENTICAL
       --> This function WORKS!
    2. After converting to matrix16, printed out matrix and checked
       by eye. It is correctly 0 padded, preserving the original matrix.
FUTURE:
    1. Add error checking if not too expensive
    2. Make independant of white space.
***********************************/
float * read_numpy_matrix16(char* path, int * dim){
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
    int nRows16 = 0;    // Number of rows in matrix16 
    int nCols16 = 0;    // Number of rows in matrix16 
    float * matrix16 = NULL;
    //float * matrix = NULL;
    float * matrix = NULL;
    FILE * f = fopen(path, "r");

    printf("reading : %s\n", path);
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
                sprintf(errStr, "ERROR!!! nchar %i != ncolsThisRow %i\n", ncols, ncolsThisRow);
                exit_with_error(errStr);
            }
            ncolsThisRow=0;
            nchar = 0;
            nline++;
        }
        nchar++;
    }
    maxchar = maxchar + 1; //+1 for null terminator?
    printf("\tdim = [nline, ncols] =  [%i, %i],  maxchar = %i \n", nline, ncols, maxchar);
    fflush(stdout);
    
    // Done with busy work - now allocate memory, read in array
    // cudaMallocManaged(&matrix, nline * maxchar * sizeof(int));
    matrix = (float*) malloc(nline * ncols * sizeof(int));
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
            //matrix[map_idx(i,j,ncols)] = (float)atof(pch);
            matrix[map_idx(i,j,ncols)] = (float)atof(pch);
            pch = strtok(NULL, " ");
            j++;
        }
        i++;
    }


    // Now fit and 0 pad into matrix whose dims are factors of 16
    // Get Dimms
    if(nline % 16 == 0){
        nRows16 = nline;
    }else{
        nRows16 = (nline / 16 + 1) * 16;
    }
    if(ncols % 16 == 0){
        nCols16= ncols;
    }else{
        nCols16 = (ncols / 16 + 1) * 16;
    }
    cudaMallocManaged(&matrix16, nRows16 * nCols16 * sizeof(float));
    // set matrix16[] = 0
    for(i=0; i<nRows16; i++){
        for(j=0; j<nCols16; j++){
            matrix16[map_idx(i,j,nCols16)] = 0;
        }
    }
    printf("\tCopying [%i %i] to 0 padded [%i %i]\n", nline,ncols,nRows16,nCols16);
    // Copy matrix to matrix16 (which is now 0 padded)
    for(i=0; i<nline; i++){
        for(j=0; j<ncols; j++){
            matrix16[map_idx(i,j,nCols16)] = matrix[map_idx(i,j,ncols)];
        }
    }
    
    /* Debug 
    for(i=0; i<nRows16; i++){
        for(j=0; j<nCols16; j++){
            printf("%*i ", 2,matrix16[map_idx(i,j,nCols16)]);
        }
        printf("\n");
    }*/
    
    free(line);
    free(entireFile);
    fclose(f);
    dim[0] = nRows16;
    dim[1] = nCols16;
    return matrix16;
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
int * reorder_row_major_as_col_major(int * B, int * dim){
    int i,j;    // Indices
    int * newM  = NULL;  //(float *)malloc(sizeof(float) * dim[0] * dim[1]); 
    gpuErrChk(cudaMallocManaged(&newM, sizeof(int) * dim[0] * dim[1]));

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
    int * A : flattened 2D array - Input array to convert
    half * B: flattened 2D array - Result
    int M   : number of Rows
    int N   : number of Cols
RETURN:
DESCRIPTION:
    Convert floating point array to half precision array
DEBUG:
    1. Spot checked B (half * array), is correct.
FUTURE:
***********************************/
__global__ void convert_float_to_half(float * A, half * B, int M, int N){
    // This is much simpler than my version //
    /*int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < M*N) {
       B[idx] = A[idx];
    }*/
    int rIdx = blockIdx.x * blockDim.x + threadIdx.x;     //Row    index
    int cIdx = blockIdx.y * blockDim.y + threadIdx.y;     //Column index
    int rStride = blockDim.x * gridDim.x;  //
    int cStride = blockDim.y * gridDim.y;  //
    int i;
    int j;
    int idx;

    for(i=rIdx; i<M; i+=rStride){
        for(j=cIdx; j<N; j+=cStride){
            idx = d_map_idx(i,j,N);
            B[idx] = __float2half_rd(A[idx]);
            //B[d_map_idx(i,j,N)] = __float2half_rd(A[d_map_idx(i,j,N)]);
            //printf("[%i %i %i %i] ::: A[%*i] = A[%*i,%*i] = %i\n", threadIdx.x,
            //       threadIdx.y, blockIdx.x, blockIdx.y, 2,d_map_idx(i,j,N), 2, i, 2, j,
            //       A[d_map_idx(i,j,N)]);
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

/* This is NVIDIA's code. See their open source license appropriately */
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;
   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
   wmma::fill_fragment(acc_frag, 0.0f);
   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;
      int bRow = i;
      int bCol = warpN * WMMA_N;
      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
   }
   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;
   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }
      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
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
        3. Taken from : 
            https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
        4. https://devblogs.nvidia.com/even-easier-introduction-cuda/
        5. If using conditionals in conditions must hold across all threads 
           in warp, otherwise, operation is likely to hang.
        6. Inspecting fragments (e.g. a_frag) is treacherous. From the c programming guide 
           "mapping of matrix elements into fragment internal storage is unspecified and
            subject to change in future architectures"
    FUTURE:
*******************************************************/
__global__ void wmma_example_ali(half * A, half * B, float * C, int M, int N, int K)
{
    // I DON'T really understand these indices - Figure out later
    int warpSize = 32;      // Don't hard code this idiot
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;     
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Diagnostic
    printf("[warpM, warpN] = [%i %i]    blockIdx = [%i %i]   threadIdx = [%i %i]\n",
           warpM, warpN, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    /************
     Fragments for Tensor operations using tensor core.

            C  =      A  *  B 
       (M x N) = (M x K) * (K x N)
    ************/
    //wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    //wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x ==1 && threadIdx.y == 1){
        printf("****************************\n\tblockDim.x = %i\n\tblockDim.y = %i\n\tgridDim.x = %i\n\tgridDim.y = %i\n\tblockIdx.x = %i\n\tblockIdx.y = %i\n\tthreadIdx.x = %i\n\tthreadIdx.y = %i\n",
               blockDim.x, blockDim.y, gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y);
    }

    for(int k=0; k<K; k+=WMMA_K){
        int aRow = warpM * WMMA_M;
        int aCol = k;   // Recall aCol is contracted with bRow
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds Checking
        if(aRow < M && aCol < K && bRow < K && bCol<N){
            // Opportunity for disaster here
            //                    (      , memory address, stride (multiple of 4 or 8))
            //wmma::load_matrix_sync(a_frag, A + aRow * M + aCol, M); // Loading in 16x16 chunk of matrix
            wmma::load_matrix_sync(a_frag, A + aRow + aCol*M, M); // Loading in 16x16 chunk of matrix
            //wmma::load_matrix_sync(b_frag, B + bRow * K + bCol, K); // Loading in 16x16 chunk of matrix
            wmma::load_matrix_sync(b_frag, B + bRow + bCol*K, K); // Loading in 16x16 chunk of matrix

            // Diagnostics
            /*printf("[warpM, warpN] = [%i %i] bIdx = [%i %i] tIdx = [%i %i] [aRow aCol bRow bCol]=[%i %i %i %i] : [%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i] x [%i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i]\n",
                    warpM, warpN, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, aRow, aCol, bRow, bCol,
                    __half2int_rd(A[aRow * M + aCol + 0]), __half2int_rd(A[aRow * M + aCol + 1]),
                    __half2int_rd(A[aRow * M + aCol + 2]), __half2int_rd(A[aRow * M + aCol + 3]),
                    __half2int_rd(A[aRow * M + aCol + 4]), __half2int_rd(A[aRow * M + aCol + 5]),
                    __half2int_rd(A[aRow * M + aCol + 6]), __half2int_rd(A[aRow * M + aCol + 7]),
                    __half2int_rd(A[aRow * M + aCol + 8]), __half2int_rd(A[aRow * M + aCol + 9]),
                    __half2int_rd(A[aRow * M + aCol +10]), __half2int_rd(A[aRow * M + aCol +11]),
                    __half2int_rd(A[aRow * M + aCol +12]), __half2int_rd(A[aRow * M + aCol +13]),
                    __half2int_rd(A[aRow * M + aCol +14]), __half2int_rd(A[aRow * M + aCol +15]),

                    __half2int_rd(B[bRow * K + bCol + 0]), __half2int_rd(B[bRow * K + bCol + 1]),
                    __half2int_rd(B[bRow * K + bCol + 2]), __half2int_rd(B[bRow * K + bCol + 3]),
                    __half2int_rd(B[bRow * K + bCol + 4]), __half2int_rd(B[bRow * K + bCol + 5]),
                    __half2int_rd(B[bRow * K + bCol + 6]), __half2int_rd(B[bRow * K + bCol + 7]),
                    __half2int_rd(B[bRow * K + bCol + 8]), __half2int_rd(B[bRow * K + bCol + 9]),
                    __half2int_rd(B[bRow * K + bCol +10]), __half2int_rd(B[bRow * K + bCol +11]),
                    __half2int_rd(B[bRow * K + bCol +12]), __half2int_rd(B[bRow * K + bCol +13]),
                    __half2int_rd(B[bRow * K + bCol +14]), __half2int_rd(B[bRow * K + bCol +15]));
            */
            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        }
    }
    // Store the output for warp
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if(cRow < M && cCol < N){
        wmma::store_matrix_sync(C + cRow + cCol * M, acc_frag, (unsigned)M, wmma::mem_row_major);
        //wmma::store_matrix_sync(C + cRow + cCol * M, acc_frag, (unsigned)M, wmma::mem_col_major);
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
    int nDev = 0;       //Number of devices
    int * dimA = NULL;  //{2,3};
    int * dimB = NULL;  //{3,2};
    int * dimAB = NULL; //{0,0};    // Initialize to some value
    float *A = NULL;
    float *B = NULL;
    half *hA = NULL;        // Converted array A to half
    half *hB = NULL;        // Converted array B to half
    float *AB = NULL;
    FILE * fout = NULL;
    dim3 blockDim;//(4,4);     // Must be used to have multidimensional grid-stride loops
    dim3 gridDim;//(2,2); 
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
        printf("maxGridSize : [%i %i %i]\n", prop.maxGridSize[0],
                prop.maxGridSize[1], prop.maxGridSize[2]);
    }


    // This uses CUDA's Unified Memory
    gpuErrChk(cudaMallocManaged(&dimA, 2 * sizeof(int)));
    gpuErrChk(cudaMallocManaged(&dimB, 2 * sizeof(int)));
    gpuErrChk(cudaMallocManaged(&dimAB, 2 * sizeof(int)));
    
    
    //sprintf(path, "data/very_small/A.txt");
    //sprintf(path, "data/smaller/A.txt");
    //sprintf(path, "data/large/A.txt");
    sprintf(path, "data/very_large/A.txt");
    A = read_numpy_matrix16(path, dimA);
    //A = reorder_row_major_as_col_major(A, dimA);
    //sprintf(path, "data/smaller/B.txt");
    //sprintf(path, "data/very_small/B.txt");
    //sprintf(path, "data/large/B.txt");
    sprintf(path, "data/very_large/B.txt");
    B = read_numpy_matrix16(path, dimB);
    time_t start = time(NULL);
    //sprintf(path, "data/AB_small.txt");
    //sprintf(path, "data/large/AB.txt");
    //B = reorder_row_major_as_col_major(B, dimB);

    // Convert from int to half
    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (dimA[0] + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (dimB[1] + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    printf("Converting array A to half precision...\n"); fflush(stdout);
    gpuErrChk(cudaMallocManaged(&hA, dimA[0] * dimA[1] * sizeof(half)));
    gpuErrChk(cudaMallocManaged(&hB, dimB[0] * dimB[1] * sizeof(half)));
    //convert_float_to_half <<<(dimA[0] * dimA[1] + 255) / 256, 256>>> (A, hA, dimA[0], dimA[1]);
    convert_float_to_half <<<gridDim,blockDim>>> (A, hA, dimA[0], dimA[1]);
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());
    printf("Converting array B to half precision...\n"); fflush(stdout);
    //convert_float_to_half <<<(dimB[0] * dimB[1] + 255) / 256, 256>>> (B, hB, dimB[0], dimB[1]);
    convert_float_to_half <<<gridDim,blockDim>>> (B, hB, dimB[0], dimB[1]);
    gpuErrChk(cudaPeekAtLastError());
    gpuErrChk(cudaDeviceSynchronize());


    // Try CUDA version of matrix_multiply
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    gpuErrChk(cudaMallocManaged(&AB, dimAB[0] * dimAB[1] * sizeof(float)));
    // Initialize to zero
    for(int i=0; i<dimAB[0] * dimAB[1]; i++)    AB[i] = 0;
    //            <<<gridDim.x (# blocks), blockDim.x (# threads per block) >>>
    printf("Multiply [%i x %i] * [%i x %i] = [%i x %i]\n", dimA[0], dimA[1], dimB[0],
           dimB[1], dimAB[0], dimAB[1]);
    fflush(stdout);
    printf("\tgridDim = [%d %d]\n\tblockDim = [%d %d]\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    //printf("\tgridDim = \n\tblockDim = \n");
    // Multiply Matrix 
    //          <<<gridD(2,2), blockD(4,4)>>>
    //blockDim.x = 128;
    //blockDim.y = 4;
    //gridDim.x = (dimA[0] + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    //gridDim.y = (dimB[1] + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    wmma_example<<<gridDim,blockDim>>> (hA, hB, AB, dimA[0], dimB[1], dimA[1], 1.0, 1.0);  
    gpuErrChk(cudaPeekAtLastError());
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

