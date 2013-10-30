#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "mpi.h"


void init_mat(double *A, double *B, double *C, int x_max, int y_max, int numtasks);
double timer();
void write_to_file(const char *txt, int x_max, int y_max, double* array);
void multiply(double* A, double* B, double* localC, char subA, char subB, int minRowC, int minColC, int dim, int numberOfProcessors, double alpha, double beta);

int main (int argc, char** argv)
{

    int numtasks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (argc != 3) {
        printf("Wrong number of parameters (%d instead of 2). Syntax:\n%s <height> <width>\n", argc - 1, argv[0]);
        exit(-1);
    }

    int x_max = atoi(argv[1]);
    int y_max = atoi(argv[2]);    
 
    if (x_max != y_max){
        printf("We consider square matrix only! %d, %d not valid.\n", x_max, y_max);
        exit(-1);
    }


    int i, rc;
    int dim = x_max;
    int arr_bytes = x_max * y_max * sizeof(double);

    double t_start, t_end, t_delta;
    t_delta = 0.0;
    double bytes = 0;
    double flops = 0;   
    double alpha = 0.1;
    double beta = 0.5;
    //Allocate matrix
    double *A = (double *) malloc(arr_bytes);
    double *B = (double *) malloc(arr_bytes);
    
    // double *localC = (double *) malloc(arr_bytes / sqrt(numtasks));
    double *localC  = (double *) malloc(arr_bytes);
    
    // for (i = 4; i < 100; i *= 4) {
        

        init_mat(A, B, localC, x_max, y_max, numtasks);


        // run matrix multiplication
        t_start = timer();
        if (rank == 1) {
            // printf("%d of %d, %d x %d matrix multiplication\n", rank, numtasks, x_max, y_max);
            multiply(A, B, localC, 'B', 'D', 0, dim / sqrt(numtasks), dim, numtasks, alpha, beta);
            multiply(A, B, localC, 'A', 'B', 0, dim / sqrt(numtasks), dim, numtasks, alpha, beta);

        } else if (rank == 2) {
            // printf("%d of %d, %d x %d matrix multiplication\n", rank, numtasks, x_max, y_max);
            multiply(A, B, localC, 'D', 'C', dim / sqrt(numtasks), 0, dim, numtasks, alpha, beta);
            multiply(A, B, localC, 'C', 'A', dim / sqrt(numtasks), 0, dim, numtasks, alpha, beta);
        
        } else if (rank == 3) {
            // printf("%d of %d, %d x %d matrix multiplication\n", rank, numtasks, x_max, y_max);
            multiply(A, B, localC, 'C', 'B', dim / sqrt(numtasks), dim / sqrt(numtasks), dim, numtasks, alpha, beta);
            multiply(A, B, localC, 'D', 'D', dim / sqrt(numtasks), dim / sqrt(numtasks), dim, numtasks, alpha, beta);

        
        } else if (rank == 0) {
            // printf("%d of %d, %d x %d matrix multiplication\n", rank, numtasks, x_max, y_max);
            multiply(A, B, localC, 'A', 'A', 0, 0, dim, numtasks, alpha, beta);
            multiply(A, B, localC, 'B', 'C', 0, 0, dim, numtasks, alpha, beta);
        }

        if (rank != 0) {
            MPI_Send(localC, x_max * y_max, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);

        } else {
            int j;
            double *C1 = (double *) malloc(arr_bytes);
            double *C2 = (double *) malloc(arr_bytes);
            double *C3 = (double *) malloc(arr_bytes);
            MPI_Status status;

            MPI_Recv(C1, x_max * y_max, MPI_DOUBLE, MPI_ANY_SOURCE, 42, MPI_COMM_WORLD, &status);
            MPI_Recv(C2, x_max * y_max, MPI_DOUBLE, MPI_ANY_SOURCE, 42, MPI_COMM_WORLD, &status);
            MPI_Recv(C3, x_max * y_max, MPI_DOUBLE, MPI_ANY_SOURCE, 42, MPI_COMM_WORLD, &status);
                // printf("%d received from %d\n", rank, status.MPI_SOURCE);
                
            for (i = 0; i < y_max; i++) {
                for (j = 0; j < x_max; j++) {
                    localC[i * x_max + j] += C1[i * x_max + j] + C2[i * x_max + j] + C3[i * x_max + j];
                }
            }

            free(C1);
            free(C2);
            free(C3);

        }
            
            // free memory space
            // free(C1);
        
        // MPI_Gather (&sendbuf,sendcnt,sendtype,&recvbuf,recvcount,recvtype,root,comm);
        /*MPI_Gather (&localC, 250 * 250, MPI_DOUBLE,
                    globalC, 250 * 250, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        */


        t_end = timer();
        t_delta = t_end - t_start;

        // statistics
        bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(double);
        flops = (double)x_max * (double)y_max * (double)x_max * 2;
        // printf("chunk: %d\t", i);
        printf("time elapsed: %lf\t", t_delta * 1.0e-9); 
        printf("gflops: %lf\t", flops / t_delta);
        printf("bandwidth: %lf\n", bytes/t_delta);

        // Write results to a file
        // if (rank == 0) {
        //     write_to_file("C.txt" , x_max, y_max, 'C');
        // }

        if (rank == 0) {
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

    #pragma omp parallel default(shared) private(a, b, c, i, j, ka, kb)
    {
        //#pragma omp for schedule(dynamic) nowait
        // #pragma omp for schedule(dynamic, chunk) nowait
        #pragma omp for schedule(runtime) nowait
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
}
