void Exchange_Borders()
{
  Debug("Exchange_Borders", 0);

  MPI_Sendrecv(........, 1, border_type[Y_DIR], ......, .....,
               ........, 1, border_type[Y_DIR], ......, .....,
               grid_comm, &status);

  MPI_Sendrecv(........, 1, border_type[Y_DIR], ......, .....,
               ........, 1, border_type[Y_DIR], ......, .....,
               grid_comm, &status);

  MPI_Sendrecv( ......., 1, border_type[X_DIR], ......, .....,
                ......., 1, border_type[X_DIR], ......, .....,
               grid_comm, &status);

  MPI_Sendrecv(........, 1, border_type[X_DIR], ......, .....,
               ........, 1, border_type[X_DIR], ......, .....,
               grid_comm, &status);
}
