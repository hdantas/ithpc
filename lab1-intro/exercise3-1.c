#include <stdio.h>
#include "mpi.h"

int np, rank;

int main(int argc, char **argv)
{
  int dest;
  float f = 1.0;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    f = 4.2;
  }
  
  MPI_Bcast(&f, 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
  printf("Node %i received %f\n", rank, f);
  
  MPI_Finalize();
  return 0;
}
