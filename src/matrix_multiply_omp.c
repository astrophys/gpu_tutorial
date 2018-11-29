/*
Author : Ali Snedden
Date   : 8/21/18
Purpose:
    This is a program that multiplies two matrices.
Debug  : 
Notes  : 
    1. To run : 
        export OMP_NUM_THREADS=20
        gcc -O3 -fopenmp src/matrix_multiply_omp.c  ### -O3 is critical
Good Weblinks:
    1. Unified Memory : https://devblogs.nvidia.com/unified-memory-cuda-beginners/

Future :
    1. Try managing memory directly on Host and Device.

*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>


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
    printf("\tdim = [nline, ncols] =  [%i, %i],  maxchar = %i \n", nline, ncols, maxchar);
    fflush(stdout);
    
    // Done with busy work - now allocate memory, read in array
    matrix = (float *)malloc(nline * maxchar * sizeof(float));
    line   = (char *)malloc(sizeof(char) * maxchar);
    i = 0;
    while(feof(f) == 0){
        if(fgets(line, maxchar, f) == NULL){
            printf("\tEnd of File Reached\n\n");
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
    int i;
    int j; 
    for(i=0; i<dim[0]; i++){
        for(j=0; j<dim[1]; j++){
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
float * omp_matrix_multiply(float * A, float * B, int * dimA, int * dimB, int * dimAB)
{
    int j = 0;          // Iterate over elements, do dot product
    int ai = 0;         // Index iterating over rows in A
    int bj = 0;         // Index iterating over columns in B
    int tid;
    int nthreads;
    float sum = 0;
    char errStr[500];
    float * result = (float *)malloc(sizeof(float) * dimA[0] * dimB[1]);

    // Error Check
    if(dimA[1] != dimB[0]){
        sprintf(errStr, "ERROR!! dimension mismatch, %i != %i", dimA[1], dimB[0]);
        exit_with_error(errStr);
    }

    #pragma omp parallel private(nthreads, tid, sum, bj, ai, j) shared(dimA, dimB, result)
    {
        tid = omp_get_thread_num();
        printf("%i / %i reporting for duty\n", tid, omp_get_num_threads());
        #pragma omp for
        for(ai=0; ai<dimA[0]; ai++){
            for(bj=0; bj<dimB[1]; bj++){
                sum = 0;
                for(j=0; j<dimA[1]; j++){
                    //printf("%.0f * %0.f\n", A[map_idx(ai, j, dimA[1])],
                    //        B[map_idx(j, bj, dimB[1])]);

                    sum += A[map_idx(ai, j, dimA[1])] * B[map_idx(j, bj, dimB[1])];
                    result[map_idx(ai,bj,dimB[1])] = sum;
                }
            }
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
int main(void)
{
    // Declare variables
    char path[100];
    int nDev = 0;      //Number of devices
    int * dimA = NULL; //{2,3};
    int * dimB = NULL; //{3,2};
    int * dimAB = NULL; //{0,0};    // Initialize to some value
    float *A = NULL;
    float *B = NULL;
    float *AB = NULL;
    float *answer = NULL;
    FILE * fout = NULL;

    dimA = (int *) malloc(2 * sizeof(int));
    dimB = (int *) malloc(2 * sizeof(int));
    dimAB= (int *) malloc(2 * sizeof(int));
    // Set dimensions
    dimA[0] = 2;
    dimA[1] = 3;
    dimB[0] = 3;
    dimB[1] = 2;
    A = (float* )malloc(dimA[0] * dimA[1] * sizeof(float));
    B = (float* )malloc(dimB[0] * dimB[1] * sizeof(float));
    
    printf("Running matrix_multiply_omp.c ...\n");
    /*
    int tid;
    int nthreads;
    int i = 0;
    int j = 0;
    //int longArray
    #pragma omp parallel private(nthreads, tid) shared(i,j)
    {
        tid = omp_get_thread_num();
        #pragma omp for
        for(i=0; i<100; i++){
            j = i*i;
            printf("[%i, %i]: tid %i / %i\n", i, j, tid, omp_get_num_threads());
        }
        #pragma omp flush

    }*/

    // Simple test
    // initialize_matrix(A,dimA,2);
    // initialize_matrix(B,dimB,4);
    // AB = omp_matrix_multiply(A, B, dimA, dimB, dimAB);

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
    
    AB = omp_matrix_multiply(A, B, dimA, dimB, dimAB);

    // Print matrices
    printf("Multiplying Trivial Matrices\n");
    printf("A (%i x %i):\n", dimA[0], dimA[1]);
    print_1D_array(A, dimA[0], dimA[1]);
    printf("B (%i x %i):\n", dimB[0], dimB[1]);
    print_1D_array(B, dimB[0], dimB[1]);
    printf("AB (%i x %i):\n", dimAB[0], dimAB[1]);
    print_1D_array(AB, dimAB[0], dimAB[1]);
    
    // Read matrix files
    free(A);
    free(B);
    free(AB);
    //sprintf(path, "data/A_small.txt");
    //sprintf(path, "data/large/A.txt");
    sprintf(path, "data/very_large/A.txt");
    A = read_numpy_matrix(path, dimA);
    //sprintf(path, "data/B_small.txt");
    //sprintf(path, "data/large/B.txt");
    sprintf(path, "data/very_large/B.txt");
    B = read_numpy_matrix(path, dimB);
    //sprintf(path, "data/AB_small.txt");
    //sprintf(path, "data/AB.txt");
    //answer = read_numpy_matrix(path, dimAB);
    //AB = cpu_matrix_multiply(A, B, dimA, dimB, dimAB);
    //print_1D_array(AB, dimAB[0], dimAB[1]);

    // Try CUDA version of matrix_multiply
    dimAB[0] = dimA[0];
    dimAB[1] = dimB[1];
    //AB = cpu_matrix_multiply(A, B, dimA, dimB, dimAB);
    fflush(stdout);
    time_t start = time(NULL);
    AB = omp_matrix_multiply(A, B, dimA, dimB, dimAB);
    printf("Run time : %.3f s\n", difftime(time(NULL), start));

    // Output
    fout = fopen("output/AB_result.txt", "w+");
    write_1D_array(AB, dimAB[0], dimAB[1], fout);
    fclose(fout);

    free(dimA);
    free(dimB);
    free(dimAB);
    free(A);
    free(B);
    free(AB);

    return 0;
}

