#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

// Number of OpenMP algorithms
#define NALGORITHMS 3

// Maximum integer value for matrix and vector
#define MAXNUMBER 100000

// Number of iterations
#define TIMES 500

// Input Size
#define NSIZE 1
#define NMAX 128
// #define NMAX 256
// #define NMAX 1447


// Seed Input
int y[NMAX];
int x[NMAX];// = {6, 2};
int A[NMAX][NMAX];// = {{1, 2}, {3, 4}};


void init(int dim){
    int i;
    for (i = 0; i < dim; i++)
        y[i] = 0;
}

void seq_function(int dim){
        /* The code for sequential algorithm */
    int i, j;

    for(i = 0; i < dim; i++) {
        y[i] = 0; 
        for(j = 0; j < dim; j++) {
            y[i] = y[i] + A[i][j] * x[j];
        }
    }
}

void omp_outer(int dim){
    int i, j;
    
    #pragma omp parallel
    {
        #pragma omp parallel for private(j)
        for(i = 0; i < dim; i++) {
            y[i] = 0; 
            for(j = 0; j < dim; j++) {
                y[i] = y[i] + A[i][j] * x[j]; 
            }
        }
    }
}

void omp_inner(int dim){
    int i, j, tmp;

    #pragma omp parallel
    {
        for(i = 0; i < dim; i++) {
            tmp = 0;
            #pragma omp parallel for reduction(+ : tmp)
            for(j = 0; j < dim; j++) {
                tmp = tmp + A[i][j] * x[j];
            }
            y[i] = tmp;
        }
    }
}

void omp_both(int dim){
    int i, j, k, out;
    int tmp[dim * dim];
    
    #pragma omp parallel
    {
        #pragma omp parallel for private(i, j, k)
        for (k = 0; k < dim * dim; k++) {
            i = (int) k / dim;
            j = k % dim;
            tmp[k] = A[i][j] * x[j];
        }

        for(i = 0; i < dim; i++) {
            out = 0;
            #pragma omp parallel for reduction(+ : out)
            for(j = 0; j < dim; j++) {
                out += tmp[i * dim + j];
            }
            y[i] = out;
        }
    }
}

int main (int argc, char *argv[])
{
    struct timeval startt, endt, result;
    int i, j, t, n, id;
    
    result.tv_sec = 0;
    result.tv_usec= 0;

    /* Generate a seed input */
    srand ( time(NULL) );
    for(i = 0; i < NMAX; i++) {
            x[i] = rand() % MAXNUMBER;
    }

    for(i = 0; i < NMAX; i++) {
        for(j = 0; j < NMAX; j++) {
            A[i][j] = rand() % MAXNUMBER;
        }
    }        

    printf("| NSize | Iterations |    Seq   |    Outer   |    Inner   |    Both    |\n");

    // for each input size
    n = NMAX;
    printf("| %5d | %10d |",n,TIMES);

    /* Run sequential algorithm */
    result.tv_usec=0;
    gettimeofday (&startt, NULL);
    for (t=0; t<TIMES; t++) {
        init(n);
        seq_function(n);
    }

    gettimeofday (&endt, NULL);
    result.tv_usec = (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
    printf(" %ld.%06ld | ", result.tv_usec/1000000, result.tv_usec%1000000);

    for(id = 0; id < NALGORITHMS; id++) {
        result.tv_sec=0; result.tv_usec=0;
        gettimeofday (&startt, NULL);
        
        for (t = 0; t < TIMES; t++)
        {
            init(n);
            if (id == 0)
                omp_outer(n);
            else if (id == 1)
                omp_inner(n);
            else if (id == 2)
                omp_both(n);
        }
        gettimeofday (&endt, NULL);
        // printResult("threaded",n);

        result.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
        printf(" %ld.%07ld | ", result.tv_usec/1000000, result.tv_usec%1000000);
    }
    printf("\n");
    return 0;
}