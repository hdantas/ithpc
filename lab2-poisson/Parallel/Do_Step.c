/* Code for the CG algorithm */ 

/* global parameters, memory allocated in InitCG*/
  double **pCG, **rCG, **vCG;
  double global_residue;

void Do_Step()
{
  int x, y;    
  double a, g, global_pdotv, pdotv, global_new_rdotr, new_rdotr;

  /* Calculate "v" in interior of my grid (matrix-vector multiply) */
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
    {
      vCG[x][y] = pCG[x][y];
      if (source[x][y] != 1)     /* only if point is not fixed  */
        vCG[x][y] -=  0.25 *(pCG[x + 1][y] + pCG[x - 1][y] +
	                     pCG[x][y + 1] + pCG[x][y - 1]) ;
    }

  pdotv = 0;
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      pdotv += pCG[x][y] * vCG[x][y];

  MPI_Allreduce(&pdotv, &global_pdotv, 1, MPI_DOUBLE, 
                MPI_SUM, grid_comm);

  a  = global_residue / global_pdotv;

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      phi[x][y] += a * pCG[x][y];

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      rCG[x][y] -= a * vCG[x][y];

  new_rdotr = 0;
  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      new_rdotr += rCG[x][y] * rCG[x][y];
 
  MPI_Allreduce(&new_rdotr, &global_new_rdotr, 1, MPI_DOUBLE, 
                MPI_SUM, grid_comm);

  g = global_new_rdotr / global_residue;
  global_residue = global_new_rdotr;

  for (x = 1; x < dim[X_DIR] - 1; x++)
    for (y = 1; y < dim[Y_DIR] - 1; y++)
      pCG[x][y] = rCG[x][y] + g * pCG[x][y];
}
