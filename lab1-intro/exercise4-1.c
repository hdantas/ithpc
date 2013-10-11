/***
* template33.c
***/
#include <stdio.h>
#include <math.h>
#include "mpi.h"

// #define N 128
#define N 8388608
#define ORIGINAL_VERSION 0
#define MODIFIED_VERSION 1


int rank, np;
int length, begin, end;
MPI_Status status;
float f[N], g[N];
double timer = 0.0;

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
  for(i = 0; i < N; i++)
  f[i] = sin(i * (1.0 / N));
}

/***
* distribute_f() distributes f[] over all nodes, where node 0 sends
* the appropriate part of f[] to each node. Every node only receives
* f[begin]...f[end].
***/
void distribute_f() {
  int dest;
  int their_begin;

  if(rank == 0) { /* Sending node */
    for(dest = 1; dest < np; dest++) {
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

  for(i = begin; i <= end; i++)
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

  if(rank == 0) { /* Receiving node */
    for(src = 1; src < np; src++) {
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
  
  for(i = 0; i < N; i++) {
    printf(" %.2f", g[i]);
  }

  printf("\n");
}

void start_timer()
{
  MPI_Barrier(MPI_COMM_WORLD);
  timer = MPI_Wtime() - timer;
}

void stop_timer()
{
  timer = MPI_Wtime() - timer;
}

void show_timer()
{
  printf("(Node %i) Timer: %.6f seconds\n", rank, timer);
}

void do_work(int version)
{
  if(rank == 0) {
    if (version == MODIFIED_VERSION)
      printf("Modified Version:\n");
    else
      printf("Original Version:\n");

    set_f();
  }

  start_timer();
  
  if (version == MODIFIED_VERSION)
    MPI_Scatter(&f[rank * length], length, MPI_FLOAT, &f[begin], length , MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    distribute_f();

  calc_g();

  if (version == MODIFIED_VERSION)
    MPI_Gather(&g[begin], length, MPI_FLOAT, &g[rank * length], length, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    collect_g();
  
  if (rank == 0) {
    stop_timer();
    show_timer();
    printf("\n");
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  init();

  do_work(ORIGINAL_VERSION);
  do_work(MODIFIED_VERSION);
 
  MPI_Finalize();
  return 0;
}
