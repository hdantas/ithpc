all: run-seq

seq: SEQ_Poisson.c
	gcc -o $@.out $^

step1: MPI_Poisson.c
	mpicc -o $@.out $^ -lm

run-seq: seq
	./$^.out

run-step1: step1
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 2

clean:
	rm -rfv *.o *.out