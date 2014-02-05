all: run-par

seq: SEQ_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

par: MPI_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

run-seq: seq
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 2 2 1
	
run-par: par
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 2 2 1

clean:
	rm -rfv *.o *.out output*.dat
