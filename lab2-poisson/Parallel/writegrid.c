#define ceildiv(a,b) (1+((a)-1)/(b))

void Write_Grid()
{
  int x, y, p;
  int grid_offs[2], grid_dim[2];
  int max_griddim[2];
  double **sub_phi;
  FILE *f;

  Debug("Write_Grid", 0);

  if (proc_rank == 0)
  {
    if ((f = fopen("output.dat", "w")) == NULL)
      Debug("Write_Grid : fopen failed", 1);

    /* allocate memory for receiving phi */
    max_griddim[X_DIR] = ceildiv(gridsize[X_DIR], P_grid[X_DIR]) + 2;
    max_griddim[Y_DIR] = ceildiv(gridsize[Y_DIR], P_grid[Y_DIR]) + 2;

    if ((sub_phi = malloc(max_griddim[X_DIR] * sizeof(*sub_phi))) == NULL)
      Debug("Write_Grid : malloc(sub_phi) failed", 1);
    if ((sub_phi[0] = malloc(max_griddim[X_DIR] * max_griddim[Y_DIR] *
                             sizeof(**sub_phi))) == NULL)
      Debug("Write_Grid : malloc(sub_phi) failed", 1);

    /* write data for process 0 to disk */
    for (x = 1; x < dim[X_DIR] - 1; x++)
      for (y = 1; y < dim[Y_DIR] - 1; y++)
        fprintf(f, "%i %i %f\n", offset[X_DIR]+x, offset[Y_DIR]+y, phi[x][y]);

    /* receive and write data form other processes */
    for (p = 1; p < P; p++)
    {
      MPI_Recv(grid_offs, 2, MPI_INT, p, 0, grid_comm, &status);
      MPI_Recv(grid_dim, 2, MPI_INT, p, 0, grid_comm, &status);
      MPI_Recv(sub_phi[0], grid_dim[X_DIR] * grid_dim[Y_DIR],
               MPI_DOUBLE, p, 0, grid_comm, &status);

      for (x = 1; x < grid_dim[X_DIR]; x++)
        sub_phi[x] = sub_phi[0] + x * grid_dim[Y_DIR];

      for (x = 1; x < grid_dim[X_DIR] - 1; x++)
        for (y = 1; y < grid_dim[Y_DIR] - 1; y++)
          fprintf(f, "%i %i %f\n", grid_offs[X_DIR]+x, grid_offs[Y_DIR]+y, sub_phi[x][y]);
    }
    free(sub_phi[0]);
    free(sub_phi);

    fclose(f);
  }
  else
  {
    MPI_Send(offset, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(dim, 2, MPI_INT, 0, 0, grid_comm);
    MPI_Send(phi[0], dim[Y_DIR] * dim[X_DIR], MPI_DOUBLE, 0, 0, grid_comm);
  }
}
