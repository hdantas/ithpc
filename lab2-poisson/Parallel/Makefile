all: run-par2

seq: SEQ_Poisson.c
	gcc -o $@.out $^

par: MPI_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

par2: Part2_MPI_Poisson.c
	mpicc -o $@.out $^ -lm -Wall

run-seq: seq
	./$^.out
	
run-par: par
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 3 3 1

run-par2: par2
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script ./$^.out 4 1 4

clean:
	rm -rfv *.o *.out output*.dat
