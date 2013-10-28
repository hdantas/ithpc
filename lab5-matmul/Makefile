DIM = 500 500
FLAGS = -lrt -lm -Wall

all: run-par

original: matmul.c
	gcc $^ -o $@.out $(FLAGS)

run-original: original
	./$^.out $(DIM)

omp: lab5.c
	gcc $^ -o $@.out $(FLAGS) -fopenmp

run-omp: omp
	./$^.out $(DIM)

par: lab5par.c
	mpicc $^ -o $@.out $(FLAGS) -fopenmp

run-par: par
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/$^.out 2 2 $(DIM)

diffA: originalA.txt A.txt
	diff $^

diffB: originalB.txt B.txt
	diff $^

diffC: originalC.txt C.txt
	diff $^	

compare: diffA diffB diffC

upload:
	scp lab5par.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make"

clean:
	rm -rfv *~ *.out

clean-all: clean
	rm -rfv A.txt B.txt C.txt originalA.txt originalB.txt originalC.txt
