void Setup_Proc_Grid(int argc, char **argv)
{
  int wrap_around[2];
  int reorder;

  Debug("My_MPI_Init", 0);

  /* Retrieve the number of processes P */
  MPI_Comm_size(......., &P);

  /* Calculate the number of processes per column and per row for the grid */
  if (argc > 2)
  {
    P_grid[X_DIR] = atoi(argv[1]);
    P_grid[Y_DIR] = atoi(argv[2]);
    if (P_grid[X_DIR] * P_grid[Y_DIR] != P)
      Debug("ERROR : Proces grid dimensions do not match with P", 1);
  }
  else
    Debug("ERROR : Wrong parameterinput", 1);

  /* Create process topology (2D grid) */
  wrap_around[X_DIR] = 0;
  wrap_around[Y_DIR] = 0;       /* do not connect first and last process */
  reorder = 1;                  /* reorder process ranks */
  MPI_Cart_create(....., ....., ....., ....., ....., &grid_comm);

  /* Retrieve new rank and carthesian coordinates of this process */
  MPI_Comm_rank(....., .....);
  MPI_Cart_coords(....., ....., ....., .....);

  printf("(%i) (x,y)=(%i,%i)\n", proc_rank, proc_coord[X_DIR], proc_coord[Y_DIR]);

  /* calculate ranks of neighbouring processes */
  MPI_Cart_shift(....., Y_DIR, ....., ....., .....);
  MPI_Cart_shift(....., X_DIR, ....., ....., .....);

  if (DEBUG)
    printf("(%i) top %i, right %i, bottom %i, left %i\n",
           proc_rank, proc_top, proc_right, proc_bottom, proc_left);
}
