#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "tbb/blocked_range.h"

// Number of OpenMP algorithms
#define NALGORITHMS 3

// Maximum integer value for matrix and vector
#define MAXNUMBER 100000

// Number of iterations
#define TIMES 500

// Input Size
#define NSIZE 1
#define NMAX 256
// #define NMAX 1447


// Seed Input
int y[NMAX];
int x[NMAX];
int A[NMAX][NMAX];

void init(int dim){
  int i, j;

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
}

void reset(int dim) {
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

void par_function(int dim){
    /* The code for parallel algorithm */
  int i, j;

  for(i = 0; i < dim; i++) {
    y[i] = 0;
    
    for(j = 0; j < dim; j++) {
      y[i] = y[i] + A[i][j] * x[j];
    }
  }
}

void print(char *txt, int dim) {
  int i;
  FILE *f = fopen(txt, "w");
      
  if (f == NULL)
  {
      printf("Error opening file!\n");
      exit(1);
  }

  for(i = 0; i < dim; i++) {
      fprintf(f, "\ty[%d] = %d\n", i, y[i]);
  }    

  fclose(f);
}

int main() {
  int dim = 4;

  init(dim);
  
  seq_function(dim);
  print("seq.txt", dim);
  
  reset(dim);
  par_function(dim);
  print("par.txt", dim);

  return 0;
}


