#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

using namespace tbb;
using namespace std;

// Maximum integer value for matrix and vector
#define MAXNUMBER 10

// Number of iterations
#define TIMES 100

// Input Size
#define NMAX 8192
#define NCOUNT 9
int NSIZE[NCOUNT] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

// Seed Input
int y[NMAX];
int x[NMAX];
int A[NMAX][NMAX];


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

void innerloop (int dim, int i) {
    int j;
    y[i] = 0;
    for(j = 0; j < dim; j++)
        y[i] = y[i] + A[i][j] * x[j];  
}

void par_function (int dim) {
    parallel_for(0, dim, 1,
        [=](int i) {
            innerloop(dim, i);
        }
    );
}

void print(const char *txt, int n) {
    int i;
    FILE *f = fopen(txt, "w");
        
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    for(i = 0; i < n; i++) {
        fprintf(f, "\ty[%d] = %d\n", i, y[i]);
    }    

    fclose(f);
}

int main (int argc, char *argv[]) {
    struct timeval startt, endt, result;
    int i, j, t, n;
    
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

    printf("| NSize | Iterations |     Seq    |     TBB    |\n");

    // for each input size
    for (i = 0; i < NCOUNT; i++) {
        n = NSIZE[i];

        printf("| %5d | %10d |",n,TIMES);

        /* Run sequential algorithm */
        result.tv_usec=0;
        gettimeofday (&startt, NULL);
        init(n);
        for (t=0; t<TIMES; t++) {
            seq_function(n);
        }
        gettimeofday (&endt, NULL);
        result.tv_usec = (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
        printf(" %2ld.%06ld  | ", result.tv_usec/1000000, result.tv_usec%1000000);
        print("seq.txt", n);

        /* Run Paralel (outer loop) algorithm */
        result.tv_sec=0; result.tv_usec=0;
        gettimeofday (&startt, NULL);
        init(n);
        for (t = 0; t < TIMES; t++) {
            par_function(n);
        }
        gettimeofday (&endt, NULL);
        result.tv_usec += (endt.tv_sec*1000000+endt.tv_usec) - (startt.tv_sec*1000000+startt.tv_usec);
        printf(" %2ld.%06ld |\n", result.tv_usec/1000000, result.tv_usec%1000000);
        print("par.txt", n);
    }

    return 0;
}