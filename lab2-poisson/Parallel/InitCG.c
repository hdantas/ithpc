void InitCG()
{
  int x, y;
  double rdotr=0;

  /* allocate memory for CG arrays*/
  pCG = malloc(dim[X_DIR] * sizeof(*pCG));
  pCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**pCG));
  for (x = 1; x < dim[X_DIR]; x++) pCG[x] = pCG[0] + x * dim[Y_DIR];

  rCG = malloc(dim[X_DIR] * sizeof(*rCG));
  rCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**rCG));
  for (x = 1; x < dim[X_DIR]; x++) rCG[x] = rCG[0] + x * dim[Y_DIR];

  vCG = malloc(dim[X_DIR] * sizeof(*vCG));
  vCG[0] = malloc(dim[X_DIR] * dim[Y_DIR] * sizeof(**vCG));
  for (x = 1; x < dim[X_DIR]; x++) vCG[x] = vCG[0] + x * dim[Y_DIR];

  /* initiate rCG and pCG */
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
    {
      rCG[x][y] = 0;
      if (source[x][y] != 1)
        rCG[x][y] = 0.25 *(phi[x + 1][y] + phi[x - 1][y] +
	                   phi[x][y + 1] + phi[x][y - 1]) - phi[x][y];
      pCG[x][y] = rCG[x][y];
      rdotr += rCG[x][y]*rCG[x][y];
    }

  /* obtain the global_residue also for the initial phi  */
  MPI_Allreduce(&rdotr, &global_residue, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
}
