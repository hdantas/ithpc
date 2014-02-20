#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"


void init_mat(double *A, double *B, double *C, int x_max, int y_max, int numtasks);
double timer();
void write_to_file(const char *txt, int x_max, int y_max, double* array);
void multiply(double* A, double* B, double* localC, char subA, char subB, int minRowC, int minColC, int dim, int numberOfProcessors, double alpha, double beta);
void readFromFile(char* filename, char* multiplyOrder);

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
    double alpha = 0.1;
    double beta = 0.5;
   
    // char multiplyOrder[numtasks][2 * sqrt(numtasks)];
    char *multiplyOrder = (char *) malloc(2 * sqrtNumtasks * numtasks * sizeof(char));
    char filename[20];
    sprintf(filename, "%d", numtasks);
    strcat(filename, ".txt");
    readFromFile(filename, multiplyOrder);


    //Allocate matrix
    double *A = (double *) malloc(arr_bytes);
    double *B = (double *) malloc(arr_bytes);
    double *C1;
    // double *localC = (double *) malloc(arr_bytes / sqrt(numtasks));
    double *localC  = (double *) malloc(arr_bytes);
    
    // for (i = 4; i < 100; i *= 4) {
        

        init_mat(A, B, localC, x_max, y_max, numtasks);


        // run matrix multiplication
        t_start = timer();
        
        t_start_cpu = timer();
        if (rank == 0) {
            for (i = 0; i < y_max; i++) { // beta * c
                k = i * x_max;
                for (j = 0; j < x_max; j++) {
                    localC[k] *= beta;
                    k++;
                }
            }
        }

        for(i = 0; i < 2 * sqrtNumtasks; i += 2) {
            minRow = steps * (rank / sqrtNumtasks);
            minCol = steps * (rank % sqrtNumtasks);
            // printf("%d computes C[%d][%d] = %c * %c\n", rank, minRow, minCol, multiplyOrder[rank * 2 * sqrtNumtasks + i], multiplyOrder[rank * 2 * sqrtNumtasks + i + 1]); 
            multiply(A, B, localC, multiplyOrder[rank * 2 * sqrtNumtasks + i], multiplyOrder[rank * 2 * sqrtNumtasks + i + 1],
                     minRow, minCol, dim, numtasks, alpha, beta);
        }
        t_end_cpu = timer();

        t_start_comm = timer();
        if (rank != 0) {
            MPI_Send(localC, x_max * y_max, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
        } else {
            MPI_Status status;
            C1 = (double *) malloc((numtasks - 1) * arr_bytes);

            for(id = 1; id < numtasks; id++) {
                MPI_Recv(&C1[(id - 1) * dimSq], x_max * y_max, MPI_DOUBLE, id, 42, MPI_COMM_WORLD, &status);
                // for (i = 0; i < y_max; i++){
                //     for (j = 0; j < x_max; j++) {
                //     printf("rank: %d, C1[%d] = %1.5lf\n", id, (id - 1) * dimSq + i * x_max + j,
                //         C1[(id - 1) * dimSq + i * x_max + j]);
                //     }
                // }
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
        // printf("chunk: %d\t", i);
        printf("rank: %2d, Total time: %lf, Comm time: %lf, CPU time: %lf", rank, t_delta * 1.0e-9, t_delta_comm * 1.0e-9, t_delta_cpu * 1.0e-9); 
        printf(", gflops: %lf", flops / t_delta);
        printf(", bandwidth: %lf\n", bytes/t_delta);

        // Write results to a file
        // if (rank == 0) {
        //     write_to_file("C.txt" , x_max, y_max, 'C');
        // }

        if (rank == 0) {
            for (i = 0; i < y_max; i++) { // alpha * AB
                k = i * x_max;
                for (j = 0; j < x_max; j++) {
                    // k = i * x_max + j;
                    for(id = 1; id < numtasks; id++) {
                        localC[k] += C1[(id - 1) * dimSq + k];
                    }
                    // printf("C[%d] = localC[%d] + C1[%d] + C1[%d] + C1[%d]\n", k, k,
                    //     k, dimSq + k, 2 * dimSq + k);
                    k++;
                }
            }
           write_to_file("C.txt" , x_max, y_max, localC);
        }

    free(A);
    free(B);
    free(localC);
    MPI_Finalize();

    return 0;
}

void init_mat(double *A, double *B, double *C, int x_max, int y_max, int numtasks)
{
    int i, j;
    for(i = 0; i < y_max; i++) {
        for(j = 0; j < x_max; j++) {
            A[i * x_max + j] = (double)i * 0.1 +  (double)j * 0.01;
            B[i * x_max + j] = (double)i * 0.01 + (double)j * 0.1;
            
            // if (i < y_max / sqrt(numtasks) && j < x_max / sqrt(numtasks))
            C[i * x_max + j] = 0.0;
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

void multiply(double* A, double* B, double* localC, char subA, char subB, int minRowC, int minColC, int dim, int numberOfProcessors, double alpha, double beta) {
    int i, j, ka, kb;
    double a, b, c;
    int numberOfLetters = sqrt(numberOfProcessors);
    int increment = dim / numberOfLetters;

    int min_rowA = ((subA - 'A') / numberOfLetters) * increment;
    int max_rowA = min_rowA + increment - 1;
    int min_colA = ((subA - 'A') % numberOfLetters) * increment;
    int max_colA = min_colA + increment - 1;

    int min_rowB = ((subB - 'A') / numberOfLetters) * increment;
    int max_rowB = min_rowB + increment - 1;
    int min_colB = ((subB - 'A') % numberOfLetters) * increment;
    int max_colB = min_colB + increment - 1;

    // #pragma omp parallel default(none) private(a, b, c, i, j, ka, kb) shared(A, B, localC, alpha, dim, min_rowA, max_rowA, min_colA, max_colA, min_colB, max_colB, min_rowB, max_rowB, minRowC, minColC)
    
        // #pragma omp for schedule(dynamic) nowait
        // #pragma omp for schedule(dynamic, chunk) nowait
        // #pragma omp for schedule(runtime) nowait
        // omp_set_num_threads(16);
        #pragma omp parallel for default(none) private(a, b, c, i, j, ka, kb) shared(A, B, localC, alpha, dim, min_rowA, max_rowA, min_colA, max_colA, min_colB, max_colB, min_rowB, max_rowB, minRowC, minColC)
        for(i = min_rowA; i <= max_rowA; i++) // rowA
        {
           for(j = min_colB; j <= max_colB; j++) // colB
           {
                c = 0.0f;
                for(ka = min_colA, kb = min_rowB; ka <= max_colA && kb <= max_rowB; ka++, kb++)
                {
                    a = A[i * dim + ka];
                    b = B[kb * dim + j];
                    c += a * b;
                    // printf("\ta = A[%d][%d] = %lf", i, ka, a);
                    // printf(", b = B[%d][%d] = %lf\n", kb, j, b);
                    // printf("\tc = %lf\n",c);
                }
                // localC[minRowC + i - min_rowA][minColC + j - min_colB] += alpha * c;
                localC[(minRowC + i - min_rowA) * dim + minColC + j - min_colB] += alpha * c;
                // printf("C[%d] = %lf\n", (minRowC + i - min_rowA) * dim + minColC + j - min_colB, localC[(minRowC + i - min_rowA) * dim + minColC + j - min_colB]);
            }
        }
    
}

void readFromFile(char* filename, char* multiplyOrder) {
    int i = 0;
    char letter;
    FILE *f = fopen(filename, "r");
    while (fscanf(f, "%c", &letter) != EOF) {
        if(letter >= 'A' && letter <= 'Z'){
            multiplyOrder[i++] = letter;
            // printf("%c\n", letter);
        }
    }
    fclose(f);
}
