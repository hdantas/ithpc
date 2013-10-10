/***
* template33.c
***/
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#define N 128

int rank, np;
int length, begin, end;
MPI_Status status;
float f[N], g[N];

/***
* init() sets the begin-index, the end-index, and the length of
* this node's part of f[] and g[].
***/
void init() {
  length = N / np;
  begin = rank * length;
  end = begin + length - 1;
}

/***
* set_f() fills f[] with initial values
***/
void set_f() {
  int i;
  for(i=0; i<N; i++)
  f[i] = sin(i * (1.0/N));
}

/***
* distribute_f() distributes f[] over all nodes, where node 0 sends
* the appropriate part of f[] to each node. Every node only receives
* f[begin]...f[end].
***/
void distribute_f() {
  int dest;
  int their_begin;

  if(rank==0) { /* Sending node */
    for(dest=1; dest<np; dest++) {
      their_begin = dest * length;
      MPI_Send(&f[their_begin], length, MPI_FLOAT, dest, 12, MPI_COMM_WORLD);
    }
  } else { /* One of the receiving nodes */
    MPI_Recv(&f[begin], length, MPI_FLOAT, 0, 12, MPI_COMM_WORLD, &status);
  }
}

/***
* calc_g() calculates g[begin]...g[end], based on f[begin]...f[end].
***/
void calc_g()
{
  int i;

  for(i=begin; i<=end; i++)
    g[i] = 2.0 * f[i];
}

/***
* collect_g() gathers the subresults from each node (g[begin]...g[end]) to
* node 0.
***/
void collect_g()
{

  int src;
  int their_begin;

  if(rank==0) { /* Receiving node */
    for(src=1; src<np; src++) {
      their_begin = src * length;
      MPI_Recv(&g[their_begin], length, MPI_FLOAT, src, 13,MPI_COMM_WORLD, &status);
    }
  } else { /* One of the sending nodes */
    MPI_Send(&g[begin], length, MPI_FLOAT, 0, 13, MPI_COMM_WORLD);
  }
}

/***
* show_g() prints g[] to the screen
***/
void show_g()
{
  int i;
  
  printf("g[]:\n");
  
  for(i=0; i<N; i++) {
    printf(" %.2f", g[i]);
  }

  printf("\n");
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  init();
  
  if(rank==0)
    set_f();

  distribute_f();
  calc_g();
  collect_g();
  
  if(rank==0)
    show_g();
  
  MPI_Finalize();
  return 0;
}