all: run_ex2-2

compile: helloworld.c
	mpicc -o helloworld.out helloworld.c

compile_ex2-2: exercise2-2.c
	mpicc -o exercise2-2.out exercise2-2.c

compile_ex2-3: exercise2-3.c
	mpicc -o exercise2-3.out exercise2-3.c

compile_ex3-1: exercise3-1.c
	mpicc -o exercise3-1.out exercise3-1.c

compile_ex3-2: exercise3-2.c
	mpicc -o exercise3-2.out exercise3-2.c -lm

compile_ex3-3: exercise3-3.c
	mpicc -o exercise3-3.out exercise3-3.c -lm

compile_ex4-1: exercise4-1.c
	mpicc -o exercise4-1.out exercise4-1.c -lm

run: compile
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/helloworld.out 2

run_ex2-2: compile_ex2-2
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise2-2.out 4

run_ex2-3: compile_ex2-3
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise2-3.out 4

run_ex3-1: compile_ex3-1
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise3-1.out 4

run_ex3-2: compile_ex3-2
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise3-2.out 4

run_ex3-3: compile_ex3-3
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise3-3.out 4

run_ex4-1: compile_ex4-1
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise4-1.out 4

load:
	module load prun
	module load mpich
	module load default-ethernet
	
clean:
	rm -rfv *.out *.o
