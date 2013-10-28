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
// double * A;
// double * B;
// double * C;
double A[500][500];
double B[500][500];
double C[500][500];

void multiply(char indexA, char indexB, int dim, int numberOfProcessors, int minRowC, int minColC);
void shiftLeft(int dim, char matrix[dim][dim], int row, int numberOfTimes);
void shiftUp(int dim, char matrix[dim][dim], int column, int numberOfTimes);
void splitWork(int numberOfProcessors, int dim);
void seq_mat_mul(int chunksize, double* A, double* B, double* C, int min_row, int max_row, int max_col);


void init_mat(int x_max, int y_max)
{
    int i, j;
    for(j = 0; j < y_max; j++)
    {
        for(i = 0; i < x_max; i++)
        {
            A[i][j] = (double)i * 0.1 +  (double)j * 0.01;
            B[i][j] = (double)i * 0.01 + (double)j * 0.1;
            C[i][j] = 0.0;
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

void write_to_file(const char *txt, int x_max, int y_max, char array) {
    int i, j;
    FILE *f = fopen(txt, "w");
        
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for(i = 0; i < y_max; i++) {
        for(j = 0; j < x_max; j++) {
            if (array == 'A')
                fprintf(f, "%s[%d] = %lf\n", txt, i * x_max + j, A[i][j]);
            else if (array == 'B')
                fprintf(f, "%s[%d] = %lf\n", txt, i * x_max + j, B[i][j]);
            else if (array == 'C')
                fprintf(f, "%s[%d] = %lf\n", txt, i * x_max + j, C[i][j]);
        }
    }    

    fclose(f);
}

/**
    main entry
**/
int main (int argc, char** argv)
{
    int i, j, k, r;
    
    // double * A;
    // double * B;
    // double * C;

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


    // for (i = 4; i < 100; i *= 4) {
        i = 500;
        x_max = 500;
        y_max = 500;
        init_mat(x_max, y_max);

        // run matrix multiplication
        t_start = timer();
        // seq_mat_mul(i, A, B, C, 1, y_max, x_max);
        printf("%d ",i);
        splitWork(4, i);
        t_end = timer();
        t_delta = t_end - t_start;

        // statistics
        bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(double);
        flops = (double)x_max * (double)y_max * (double)x_max * 2;
        printf("chunk: %d\t",i);
        printf("time elapsed: %lf\t", t_delta*1.0e-9); 
        printf("gflops: %lf\t", flops/t_delta);
        printf("bandwidth: %lf\n", bytes/t_delta);
    // }

    write_to_file("A.txt", x_max, y_max, 'A');
    write_to_file("B.txt", x_max, y_max, 'B');
    write_to_file("C.txt", x_max, y_max, 'C');


    return 0;
}

void splitWork(int numberOfProcessors, int dim) { //matrices are square
    int i, j, k, steps, size;
    int increment;
    double result[dim][dim];
    
    steps = sqrt(numberOfProcessors);
    increment = dim / steps;
    char orderA[steps][steps];
    char orderB[steps][steps];
    char orderC[steps][steps][2 * steps];

    // 1. Partition these matrices in square blocks p, where p is the number of processes available.
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            orderA[i][j] = 'A' + i * steps + j;
            orderB[i][j] = 'A' + i * steps + j;
        }
    }
    
    // 2. Create a matrix of processes of size p 1/2 x p 1/2 so that each process can maintain a block of A matrix and a block of B matrix.
    //init MPI

    // initial alignment
    for (i = 0; i < steps; i++) {
        shiftLeft(steps, orderA, i, i);
        shiftUp(steps, orderB, i, i);
    }


    // 5. Repeat steps 3 y 4 sqrt(p) times.
    for (k = 0; k < steps; k++) {
        for (i = 0; i < steps; i++) {
            for (j = 0; j < steps; j++) {
                orderC[i][j][2 * k + 0] = orderA[i][j];
                orderC[i][j][2 * k + 1] = orderB[i][j];
                // printf("orderC[%d][%d] = %c * %c\n",i, j, orderC[i][j][2 * k + 0], orderC[i][j][2 * k + 1]);
            }
        }

        for (i = 0; i < steps && k < steps - 1; i++) { //does not need to run in the last iteration of the for-k loop
            shiftLeft(steps, orderA, i, 1);
            shiftUp(steps, orderB, i, 1);
        }
    }
    // printf("done\n");
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            // printf("C[%d][%d] = ", i, j);
            for (k = 0; k < steps; k++) {
                // printf("%c * %c + ", orderC[i][j][2 * k], orderC[i][j][2 * k + 1]);
                multiply(orderC[i][j][2 * k], orderC[i][j][2 * k + 1], dim, numberOfProcessors, i * increment, j * increment); //verify
            }
            // printf("\n");
        }
    }

}

void multiply(char subA, char subB, int dim, int numberOfProcessors, int minRowC, int minColC ) {
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

    for(i = min_rowA; i <= max_rowA; i++) // rowA
    {
       for(j = min_colB; j <= max_colB; j++) // colB
       {
            c = 0.0f;
            for(ka = min_colA, kb = min_rowB; ka <= max_colA && kb <= max_rowB; ka++, kb++)
            {
                a = A[i][ka];
                b = B[kb][j];
                c += a * b;
                // printf("\ta = A[%d][%d] = %lf", i, ka, a);
                // printf(", b = B[%d][%d] = %lf\n", kb, j, b);
                // printf("\tc = %lf\n",c);
            }
            C[minRowC + i - min_rowA][minColC + j - min_colB] += alpha * c;
            // printf("C[%d][%d] = %lf\n", minRowC + i - min_rowA, minColC + j - min_colB, C[minRowC + i - min_rowA][minColC + j - min_colB]);
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

void shiftLeft(int dim, char matrix[dim][dim], int row, int numberOfTimes) {
    int j, k;
    char tmp;
    for(k = 0; k < numberOfTimes; k++) {
        tmp = matrix[row][0];
        for(j = 1; j < dim; j++) {
            matrix[row][j - 1] = matrix[row][j];
        }
        matrix[row][dim - 1] = tmp;
    }
}


void shiftUp(int dim, char matrix[dim][dim], int column, int numberOfTimes) {
    int i, k;
    char tmp;
    for(k = 0; k < numberOfTimes; k++) {
        tmp = matrix[0][column];
        for(i = 1; i < dim; i++) {
            matrix[i - 1][column] = matrix[i][column];
        }
        matrix[dim - 1][column] = tmp;
    }
}