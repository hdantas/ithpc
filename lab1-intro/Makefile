all: run

compile: helloworld.c
	mpicc -o helloworld helloworld.c

run: compile
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/helloworld 2

load:
	module load prun
	module load mpich
	module load default-ethernet
	
clean:
	rm -rfv *.out
