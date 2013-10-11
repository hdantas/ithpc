void Debug(char *mesg, int terminate)
{
  if (DEBUG || terminate)
    printf("(%i) %s\n", proc_rank, mesg);
  if (terminate)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
  }
}

 
