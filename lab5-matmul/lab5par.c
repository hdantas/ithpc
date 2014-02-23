#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define ALPHA 0.1
#define BETA 0.5
#define BLOCK_SIZE 36

void init_mat(double *A, double *B, double *C, int x_max, int y_max, int numtasks, int rank);
double timer();
void write_to_file(const char *txt, int x_max, int y_max, double* array);
void multiply(double* A, double* B, double* C, char subA, char subB, int minRowC, int minColC, int dim, int numberOfProcessors);
void readFromFile(char* filename, char* multiplyOrder);
void matrixMul_block_par (double* C, double* A, double* B, int numberOfProcessors, int dim, 
    int minRowA, int minColA, int minRowB, int minColB, int minRowC, int minColC);

int main (int argc, char** argv)
{

    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (argc != 3) {
        printf("Wrong number of parameters (%d instead of 2). Syntax:\n%s <height> <width>\n", argc - 1, argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    int x_max = atoi(argv[1]);
    int y_max = atoi(argv[2]);    
 
    if (x_max != y_max){
        printf("We consider a square matrix only! %d, %d not valid.\n", x_max, y_max);
        MPI_Finalize();
        exit(-1);
    }

    int sqrtNumtasks = (int) sqrt(numtasks);
    if (sqrtNumtasks != sqrt(numtasks)) {
        printf("The number of processors must be a square of an integer.\n");
        MPI_Finalize();
        exit(-1);
    }

    if (x_max % sqrtNumtasks != 0) {
        printf("The matrix dimensions must be divisible by the square root of the number of processors.\n");
        MPI_Finalize();
        exit(-1);
    }

    
    int i, j, k, id;
    int dim = x_max;
    int arr_bytes = x_max * y_max * sizeof(double);
    int minRow, minCol;
    int dimSq = dim * dim;
    int steps = dim / sqrtNumtasks;
    double t_start, t_end, t_delta;
    double t_start_comm, t_end_comm, t_delta_comm;
    double t_start_cpu, t_end_cpu, t_delta_cpu;
    t_delta = t_delta_comm = t_start_cpu = 0.0;
    
    double bytes = 0;
    double flops = 0;   
   
    char *multiplyOrder = (char *) malloc(2 * sqrtNumtasks * numtasks * sizeof(char));
    char filename[20];
    sprintf(filename, "%d", numtasks);
    strcat(filename, ".txt");
    readFromFile(filename, multiplyOrder);


    //Allocate matrix
    double *A = (double *) malloc(arr_bytes);
    double *B = (double *) malloc(arr_bytes);
    double *C = (double *) malloc(arr_bytes);
    
    double *tempC;
        
    init_mat(A, B, C, x_max, y_max, numtasks, rank);
    
    t_start = timer();
    t_start_cpu = timer();

    for(i = 0; i < 2 * sqrtNumtasks; i += 2) {
        minRow = steps * (rank / sqrtNumtasks);
        minCol = steps * (rank % sqrtNumtasks);
        multiply(A, B, C, multiplyOrder[rank * 2 * sqrtNumtasks + i], multiplyOrder[rank * 2 * sqrtNumtasks + i + 1], minRow, minCol, dim, numtasks);
    }
    t_end_cpu = timer();
    
    t_start_comm = timer();
    if (rank != 0) {
        MPI_Send(C, x_max * y_max, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        tempC = (double *) malloc((numtasks - 1) * arr_bytes);

        for(id = 1; id < numtasks; id++) {
            MPI_Recv(&tempC[(id - 1) * dimSq], x_max * y_max, MPI_DOUBLE, id, 42, MPI_COMM_WORLD, &status);
        }

    }
    t_end_comm = timer();
        
    t_end = timer();

    t_delta_comm = t_end_comm - t_start_comm;
    t_delta_cpu = t_end_cpu - t_start_cpu;
    t_delta = t_end - t_start;

    // statistics
    bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(double);
    flops = (double)x_max * (double)y_max * (double)x_max * 2;
    
    printf("rank: %2d, Total time: %lf, Comm time: %lf, CPU time: %lf", rank, t_delta * 1.0e-9, t_delta_comm * 1.0e-9, t_delta_cpu * 1.0e-9); 
    printf(", gflops: %lf", flops / t_delta);
    printf(", bandwidth: %lf\n", bytes/t_delta);

    if (rank == 0) {
        for (i = 0; i < y_max; i++) {
            k = i * x_max;
            for (j = 0; j < x_max; j++) {
                for(id = 1; id < numtasks; id++) {
                    C[k] += tempC[(id - 1) * dimSq + k];
                }
                C[k] += BETA * ((double)j * 0.11);
                k++;
            }
        }
       // write_to_file("C.txt" , x_max, y_max, C);
       free(tempC);
    }


    free(A);
    free(B);
    free(C);
    
    MPI_Finalize();

    return 0;
}

void init_mat(double *A, double *B, double *C, int x_max, int y_max, int numtasks, int rank)
{
    int i, j;
    for(i = 0; i < y_max; i++) {
        for(j = 0; j < x_max; j++) {
            A[i * x_max + j] = (double)i * 0.1 +  (double)j * 0.01;
            B[i * x_max + j] = (double)i * 0.01 + (double)j * 0.1;
            C[i * x_max + j] = 0;
        }
    }
}

double timer()
{
    long t_val = 0; 
    struct timespec ts;
    
    clock_gettime(CLOCK_REALTIME,&ts);
    t_val = ts.tv_sec * 1e+9 + ts.tv_nsec;
        
    return (double)t_val;   
}

void write_to_file(const char *txt, int x_max, int y_max, double* array) {
    int i, j;
    FILE *f = fopen(txt, "w");
        
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for(i = 0; i < y_max; i++) {
        for(j = 0; j < x_max; j++) {
            fprintf(f, "%s[%d] = %lf\n", txt, i * x_max + j, array[i * x_max + j]);
        }
    }    

    fclose(f);
}

void multiply(double* A, double* B, double* C, char subA, char subB, int min_rowC, int min_colC, int dim, int numberOfProcessors) {
    int numberOfLetters = sqrt(numberOfProcessors);
    int increment = dim / numberOfLetters;

    int min_rowA = ((subA - 'A') / numberOfLetters) * increment;
    int min_colA = ((subA - 'A') % numberOfLetters) * increment;

    int min_rowB = ((subB - 'A') / numberOfLetters) * increment;
    int min_colB = ((subB - 'A') % numberOfLetters) * increment;

    matrixMul_block_par (C, A, B, numberOfProcessors, dim, min_rowA, min_colA, min_rowB, min_colB, min_rowC, min_colC);
    
}

// Matrix multiplication on the device C = A * B
void matrixMul_block_par (double* C, double* A, double* B, int numberOfProcessors, int dim, 
    int minRowA, int minColA, int minRowB, int minColB, int minRowC, int minColC)
{
    double As[BLOCK_SIZE][BLOCK_SIZE];
    double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx, by; // Block index
    int tx, ty;
    int c, k;
    double Asub, Bsub, Csub;

    int sub_matrix_dim = dim / sqrt(numberOfProcessors); // submatrix assign to this processor
    int subgrid_dim = (sub_matrix_dim / BLOCK_SIZE);     // number of subgrids. Dimensions of the subgrid are (BLOCK_SIZE, BLOCK_SIZE).
    int hA_grid = subgrid_dim;                           // number of A subgrids along the y axis
    int wB_grid = subgrid_dim;                           // number of B subgrids along the x axis

    int tmp_indexA, tmp_indexB, tmp_indexC;
    unsigned int i;
    
    #pragma omp parallel for default(none) \
        shared(A, B, C, dim, minColA, minRowA, minRowB, minColB, minColC, minRowC, hA_grid, wB_grid, subgrid_dim) \
        private(As, Bs, bx, by, i, k, c, ty, tx, Asub, Bsub, Csub, tmp_indexA, tmp_indexB, tmp_indexC)
    for (by = 0; by < hA_grid; by++) { // go through all the subgrids for this processor
        for (bx = 0; bx < wB_grid; bx++) {

            // Loop over all the subsubgrid operations required to compute the subgrid
            for (i = 0; i < subgrid_dim; i++) { // travel on the subsubgrid
                for (ty = 0; ty < BLOCK_SIZE; ty++) {
                    for (tx = 0; tx < BLOCK_SIZE; tx++) {
                        tmp_indexA = i * BLOCK_SIZE + by * BLOCK_SIZE * dim + dim * ty + tx + minColA + minRowA * dim;
                        tmp_indexB = i * BLOCK_SIZE * dim + bx * BLOCK_SIZE + dim * ty + tx + minColB + minRowB * dim;

                        As[ty][tx] = A[tmp_indexA];
                        Bs[ty][tx] = B[tmp_indexB];
                    }// for tx
                }// for ty

                // Multiply the two matrices togethers                         
                for (ty = 0; ty < BLOCK_SIZE; ty++) {
                    for (tx = 0; tx < BLOCK_SIZE; tx++) {
                        Csub = 0.0;
                        for (k = 0; k < BLOCK_SIZE; ++k) {
                            Asub = As[ty][k ];
                            Bsub = Bs[k ][tx];
                            Csub += Asub * Bsub;
                        }
                        c = dim * BLOCK_SIZE * by + BLOCK_SIZE * bx;
                        tmp_indexC = c + dim * ty + tx + minColC + minRowC * dim;
                        C[tmp_indexC] += (double) ALPHA * Csub;
                        // printf("C[%d] = %lf\n", tmp_indexC, C[tmp_indexC]);
                    }// for tx
                }// for ty
            }// for subsubgrids
        
        }// for bx
    }// for by

}

void readFromFile(char* filename, char* multiplyOrder) {
    int i = 0;
    char letter;
    FILE *f = fopen(filename, "r");
    while (fscanf(f, "%c", &letter) != EOF) {
        if(letter >= 'A' && letter <= 'Z'){
            multiplyOrder[i++] = letter;
        }
    }
    fclose(f);
}
