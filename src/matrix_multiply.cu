/*
Author : Ali Snedden
Date   : 8/21/18
Purpose:
    This is a program that multiplies two matrices.
Debug  : 
Notes  : 
    1. How to Run:
        module load cuda/8.0 
        nvcc matrix_multiply.cu 
    2. http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/docs/NVIDIA_CUDA_C_ProgrammingGuide_3.1.pdf

Future :


*/
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime_api.h>

/**********************************
ARGS:
RETURN:
DESCRIPTION:
    Map 2D indices to 1D index
DEBUG:
FUTURE:
    1. Add error checking if not too expensive
***********************************/
int map_idx(int i, int j, int Ny){
    return (Ny * i + j);
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
            printf("%*.0f ", 3, array1D[idx]);
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
    FUTURE:
*******************************************************/
void exit_with_error(char * message){
    fprintf(stderr, "%s", message);
    fflush(stderr);
    exit(1);
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
            A[i*dim[0]+j] = value;
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
    //float ** result = (float **)malloc(sizeof(float *) * dimA[0]);
    // rows
    //for(i=0; i<dimA[0]; i++){
    //    // columns
    //    result[i] = (float *)malloc(sizeof(float) * dimB[1]);
    //    for(j=0; j<dimB[1]; j++){
    //        result[i,j] = 0;
    //    }
    //}

    // Error Check
    if(dimA[1] != dimB[0]){
        sprintf(errStr, "ERROR!! dimension mismatch, %i != %i", dimA[1], dimB[0]);
        exit_with_error(errStr);
    }

    for(ai=0; ai<dimA[0]; ai++){
        for(bj=0; bj<dimB[1]; bj++){
            sum = 0;
            for(j=0; j<dimA[1]; j++){
                printf("%.0f * %0.f\n", A[map_idx(ai, j, dimA[1])],
                        B[map_idx(j, bj, dimB[1])]);

                sum += A[map_idx(ai, j, dimA[1])] * B[map_idx(j, bj, dimB[1])];
                result[map_idx(ai,bj,dimB[1])] = sum;
            }
            printf("\n");
        }
    }
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    return result;
}


/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
__global__ void matrix_multiply(float * A, float * B, int * dimA, int * dimB)
{



}





/********************************************************
    ARGS:
    DESCRIPTION:
    RETURN:
    DEBUG:
    NOTES: 
    FUTURE:
*******************************************************/
int main(void)
{
    // Declare variables
    time_t start = time(NULL);
    int dimA[] = {2,3};
    int dimB[] = {3,2};
    int dimAB[] = {0,0};    // Initialize to some value
    float *A = NULL;
    float *B = NULL;
    float *AB = NULL;
    cudaMallocManaged(&A, dimA[0] * dimA[1] * sizeof(float));
    cudaMallocManaged(&B, dimB[0] * dimB[1] * sizeof(float));

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
    
    printf("A (%i x %i):\n", dimA[0], dimA[1]);
    print_1D_array(A, dimA[0], dimA[1]);
    printf("B (%i x %i):\n", dimB[0], dimB[1]);
    print_1D_array(B, dimB[0], dimB[1]);

    AB = cpu_matrix_multiply(A, B, dimA, dimB, dimAB);
    printf("AB (%i x %i):\n", dimAB[0], dimAB[1]);
    print_1D_array(AB, dimAB[0], dimAB[1]);
    




    printf("Run time : %.3f s\n", difftime(time(NULL), start));
    return 0;
}

