#include "mpi.h"
#include <stdio.h>
#define SIZE 4

int main(argc,argv)
  int argc;
  char *argv[];
{
  
  int numtasks, rank, sendcount, recvcount, source;
  int i;

  float sendbuf[SIZE][SIZE] = {{0.0}};
  float recvbuf[SIZE];

  printf("argc = %d\n\n", argc);
  for (i = 0; i < argc; i++) {
    printf("argv[%d]: %s\n", i, argv[i]);
  }
  printf("\n");
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  if(rank == 3) {
    for (i = 0; i < SIZE * SIZE; i++) {
      sendbuf[i / SIZE][i % SIZE] = i + 1;
    }
  }

  if (numtasks == SIZE) {
    source = 3;
    sendcount = SIZE;
    recvcount = SIZE;
    MPI_Scatter(sendbuf,sendcount,MPI_FLOAT,recvbuf,recvcount,
               MPI_FLOAT,source,MPI_COMM_WORLD);

    for (i = 0; i < SIZE; i++){
      recvbuf[i] *= 2;
    }

    printf("rank= %d  Results: %.6f\t%.6f\t%.6f\t%.6f\n",rank,recvbuf[0],
           recvbuf[1],recvbuf[2],recvbuf[3]);
  } else
    printf("Must specify %d processors. Specified %d. Terminating.\n", SIZE, numtasks);

  MPI_Finalize();
}