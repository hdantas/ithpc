#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <math.h>
//#define MIC
// #define CHUNKSIZE 100

double alpha = 0.1;
double beta = 0.5;

void splitWork(int numberOfProcessors, int dim);
void seq_mat_mul(int chunksize, double* A, double* B, double* C, int 
min_row, int max_row, int max_col);


void init_mat(double *  A, double *  B, double *  C, int x_max, int y_max)
{
    int i,j;
    for(j=0; j<y_max; j++)
    {
        for(i=0; i<x_max; i++)
        {
            A[i+j*x_max] = (double)j*0.1 + (double)i*0.01;
            B[i+j*x_max] = (double)j*0.01 + (double)i*0.1;
            C[i+j*x_max] = 0.0;
        }
    }
}

/**
    timer
**/
double timer()
{
    long t_val = 0; 
    struct timespec ts;
    
    clock_gettime(CLOCK_REALTIME,&ts);
    t_val = ts.tv_sec * 1e+9 + ts.tv_nsec;
        
    return (double)t_val;   
}

void write_to_file(const char *txt, int n, double* array) {
    int i;
    FILE *f = fopen(txt, "w");
        
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for(i = 0; i < n; i++) {
        fprintf(f, "%s[%d] = %lf\n", txt, i, array[i]);
    }    

    fclose(f);
}

/**
    main entry
**/
int main (int argc, char** argv)
{
    int i, j, k, r;
    
    double * A;
    double * B;
    double * C;

    if (argc != 3)
    {
        printf("Wrong number of parameters. Syntax:\n%s <height> <width>\n", argv[0]);
        exit(-1);
    }

    int x_max = atoi(argv[1]);
    int y_max = atoi(argv[2]);  
    
    if(x_max != y_max){
        printf("we consider square matrix only!\n");
        exit(-1);
    }

    int arr_bytes = x_max * y_max * sizeof(double);
    int s = x_max;
    int length = s * s;
    double t_start, t_end, t_delta;
    t_delta = 0.0;
    double bytes = 0;
    double flops = 0;
    
    int delta_x;
    int nodes = 4;
    int min_row = 0;
    int max_row = y_max;
        
    // allocate matrix
    A = (double * )malloc(arr_bytes);
    B = (double * )malloc(arr_bytes);
    C = (double * )malloc(arr_bytes);

    splitWork(16, 4);
    return;

    for (i = 1; i < 100000; i *= 2) {
        init_mat(A, B, C, x_max, y_max);        

        // run matrix multiplication
        t_start = timer();
        seq_mat_mul(i, A, B, C, 1, y_max, x_max);
        t_end = timer();
        t_delta = t_end - t_start;

        // statistics
        bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(double);
        flops = (double)x_max * (double)y_max * (double)x_max * 2;
        printf("chunk: %d\t",i);
        printf("time elapsed: %lf\t", t_delta*1.0e-9); 
        printf("gflops: %lf\t", flops/t_delta);
        printf("bandwidth: %lf\n", bytes/t_delta);
    }

    write_to_file("A.txt", x_max * y_max, A);
    write_to_file("B.txt", x_max * y_max, B);
    write_to_file("C.txt", x_max * y_max, C);

    // free memory space
    free(A);
    free(B);
    free(C);

    return 0;
}

void splitWork(int numberOfProcessors, int dim) { //matrices are square
    int i, j, k, steps, size;
    steps = sqrt(numberOfProcessors);
    char orderA[steps][steps];
    char orderB[steps][steps];
    char orderC[steps][steps][2 * steps];

    // 1. Partition these matrices in square blocks p, where p is the number of processes available.
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            orderA[i][j] = 'A' + i * steps + j;
            orderB[i][j] = 'A' + i * steps + j;
            printf("orderA[%d][%d] = %c\n", i, j, orderA[i][j]);
            printf("orderB[%d][%d] = %c\n", i, j, orderB[i][j]);
        }
    }
    
// 2. Create a matrix of processes of size p 1/2 x p 1/2 so that each process can maintain a block of A matrix and a block of B matrix.
    //init MPI
    printf("Reodering %d times!\n", steps);

    char tmpA, tmpB;
// 5. Repeat steps 3 y 4 sqrt(p) times.
    for (k = 0; k < steps; k++) {
        for (i = 0; i <= k; i++) {
            
            tmpA = orderA[steps - i - 1][0]; //shift row left starting from bottom
            tmpB = orderB[0][steps - i - 1]; //shift column up starting from rightmost
            // printf("\ttmpA : %c, tmpB: %c\n",tmpA, tmpB);
            for (j = 1; j < steps; j++) {
            // 3. Each block is sent to each process, and the copied sub blocks are multiplied together and the results added to the partial results in the C sub-blocks.
            // multiply();
            
            // 4. The A sub-blocks are rolled one step to the left and the B sub-blocks are rolled one step upward.
                orderA[steps - i - 1][j - 1] = orderA[steps - i - 1][j];
                orderB[j - 1][steps - i - 1] = orderB[j][steps - i - 1];
            }

            orderA[steps - i - 1][steps - 1] = tmpA;
            orderB[steps - 1][steps - i - 1] = tmpB;

        }  
        // printf("After %d shift\n", k + 1);
        for (i = 0; i < steps; i++) {
            for (j = 0; j < steps; j++) {
                // printf("orderA[%d][%d] = %c\n", i, j, orderA[i][j]);
                // printf("orderB[%d][%d] = %c\n", i, j, orderB[i][j]);
                orderC[i][j][2 * k + 0] = orderA[i][j];
                orderC[i][j][2 * k + 1] = orderB[i][j];
                // printf("C[%d][%d] = %c * %c\n", i, j, orderC[i][j][0], orderC[i][j][1]);
            }
        }
        // printf("\n");
    }
    // printf("done\n");
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            printf("C[%d][%d] = ", i, j);
            for (k = 0; k < steps; k++) {    
                printf("%c * %c", orderC[i][j][2 * k + 0], orderC[i][j][2 * k + 1]);
                if (k != steps -1)
                    printf(" + ");
            }
            printf("\n");
        }
    }

}

void seq_mat_mul(int chunksize, double* A, double* B, double* C, int min_row, int max_row, int max_col) {
    
    int i, j, k;
    double a, b, c;
    int chunk = chunksize;

    #pragma omp parallel shared(alpha, beta, A, B, C, chunk, max_col, max_row) private(a, b, c, i, j, k)
    {
        //#pragma omp for schedule(dynamic) nowait
        #pragma omp for schedule(dynamic, chunk) nowait
        //#pragma omp for schedule(static) nowait
        for(j = min_row - 1; j < max_row; j++) // row_
        {
           for(i = 0; i < max_col; i++) // col_
           {
                c = 0.0f;
                for(k = 0; k < max_col; k++)
                {
                    a = A[k + j * max_col];
                    b = B[i + k * max_col];
                    c += a * b;
                    // printf("\ta = A[%d] = %lf\n", k + j * max_col, A[k + j * max_col]);
                    // printf("\tb = B[%d] = %lf\n", i + k * max_col, B[i + k * max_col]);
                    // printf("\tc = %lf\n",c);
                }
                C[i + j * max_col] = beta * C[i + j * max_col] + alpha * c;
                // printf("C[%d] = %lf\n", i + j * max_col, C[i + j * max_col]);
            }
        }
    }

}
