all: run-seq

seq: SEQ_Poisson.c
	gcc -o $@.out $^ -Wall

step1: MPI_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

run-seq: seq
	./$^.out

run-step1: step1
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 16 4 4

clean:
	rm -rfv *.o *.out
