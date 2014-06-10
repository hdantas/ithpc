all: run-par2

seq: SEQ_Poisson.c
	gcc -o $@.out $^

par: MPI_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

run-seq: seq
	./$^.out
	
run-par: par
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 3 3 1

clean:
	rm -rfv *.o *.out output*.dat
