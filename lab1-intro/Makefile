all: run_ex2-2

compile: helloworld.c
	mpicc -o helloworld helloworld.c

compile_ex2-2: exercise2-2.c
	mpicc -o exercise2-2 exercise2-2.c

compile_ex2-3: exercise2-3.c
	mpicc -o exercise2-3 exercise2-3.c

compile_ex3-1: exercise3-1.c
	mpicc -o exercise3-1 exercise3-1.c

run: compile
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/helloworld 2

run_ex2-2: compile_ex2-2
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise2-2 4

run_ex2-3: compile_ex2-3
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise2-3 4

run_ex3-1: compile_ex3-1
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/exercise3-1 4

load:
	module load prun
	module load mpich
	module load default-ethernet
	
clean:
	rm -rfv *.out
