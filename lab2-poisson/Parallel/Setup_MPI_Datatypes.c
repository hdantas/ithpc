void Setup_MPI_Datatypes()
{
  Debug("Setup_MPI_Datatypes", 0);

  /* Datatype for vertical data exchange; exchange in y-direction */
  MPI_Type_vector(....., 1, .....,
                  ....., &border_type[Y_DIR]);
  MPI_Type_commit(.........);

  /* Datatype for horizontal data exchange; exchange in x-direction */
  MPI_Type_vector(....., 1, .....,
                  ....., &border_type[X_DIR]);
  MPI_Type_commit(.........);
}
