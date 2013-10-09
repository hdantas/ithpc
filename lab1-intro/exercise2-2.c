#include <stdio.h>
#include "mpi.h"

int np, rank;

int main(int argc, char **argv)
{
  int source;
  float f = 1.0;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    for (source = 1; source <= np; source++) {
      MPI_Recv(&f, 1, MPI_FLOAT, source, 42, MPI_COMM_WORLD, &status);
      printf("%f from node %i\n", f, source);
    }
  } else {
    f *= f;
    MPI_Send(&f, 1, MPI_FLOAT, 0, 42, MPI_COMM_WORLD);
    f++;
  }
  
  MPI_Finalize();
  return 0;
}
