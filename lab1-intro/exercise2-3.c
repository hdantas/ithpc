#include <stdio.h>
#include "mpi.h"

int np, rank;

int main(int argc, char **argv)
{
  int dest;
  float f = 4.2;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    for (dest = 1; dest < np; dest++) {
      f *= f;
      MPI_Send(&f, 1, MPI_FLOAT, dest, 42, MPI_COMM_WORLD);
      f++;
    }
  } else {
      MPI_Recv(&f, 1, MPI_FLOAT, 0, 42, MPI_COMM_WORLD, &status);
      printf("Node %i received %f\n", rank, f);
  }
  
  MPI_Finalize();
  return 0;
}
