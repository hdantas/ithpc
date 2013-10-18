#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <math.h>
//#define MIC
#define CHUNKSIZE 100

double alpha = 0.1;
double beta = 0.5;

void seq_mat_mul(double* A, double* B, double* C, int min_row, int max_row, int max_col);


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
    int i,j,k,r;
    
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

    init_mat(A, B, C, x_max, y_max);        


    // run matrix multiplication
    t_start = timer();
    /*for (i = 1; i <= nodes; i++) {
        min_row = (i - 1) * y_max / nodes + 1;
        max_row = i * y_max / nodes;
        seq_mat_mul(A, B, C, min_row, max_row, x_max);
        // printf("seq_mat_mul(A, B, C, %d, %d, %d);\n", min_row, max_row, x_max);
    }*/

    seq_mat_mul(A, B, C, 1, y_max, x_max);

    t_end = timer();
    t_delta = t_end - t_start;
    
    write_to_file("A.txt", x_max * y_max, A);
    write_to_file("B.txt", x_max * y_max, B);
    write_to_file("C.txt", x_max * y_max, C);


    // statistics
    bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(double);
    flops = (double)x_max * (double)y_max * (double)x_max * 2;

    printf("time elapsed: %lf\n", t_delta*1.0e-9); 
    printf("gflops: %lf\t\n", flops/t_delta);
    printf("bandwidth: %lf\t\n", bytes/t_delta);

    // free memory space
    free(A);
    free(B);
    free(C);

    return 0;
}


void seq_mat_mul(double* A, double* B, double* C, int min_row, int max_row, int max_col) {
    
    int i, j, k;
    double a, b, c;
    int chunk = CHUNKSIZE;

    #pragma omp parallel shared(a, b, c, chunk, max_col, max_row) private(i, j, k) {

        #pragma omp for schedule(dynamic,chunk) nowait
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