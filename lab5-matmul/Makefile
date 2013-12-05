DIM = 64 64
P = 4
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
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/$^.out $(P) $(DIM)

mpi: lab5mpi.c
	mpicc $^ -o $@.out $(FLAGS) -fopenmp

run-mpi: mpi
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/$^.out $(P) $(DIM)

block: mmul_block.c
	gcc $^ -o $@.out $(FLAGS) -fopenmp

run-block: block
	./$^.out

writeFile: writeFile.c
	gcc $^ -o $@.out $(FLAGS)

run-writeFile: writeFile
	./$^.out $(P)	

diffA: originalA.txt A.txt
	diff $^

diffB: originalB.txt B.txt
	diff $^

diffC: originalC.txt C.txt
	diff $^	

diff: diffA diffB diffC

remote-mpi:
	scp lab5mpi.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make run-mpi"

remote-omp:
	scp lab5.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make run-omp"

remote-original:
	scp matmul.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make run-original"

test:
	scp mmul_block.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make run-block"

download:
	scp in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul/C.txt ~/repos/ithpc/lab5-matmul/txts/C.txt

clean:
	rm -rfv *~ *.out

clean-all: clean
	rm -rfv A.txt B.txt C.txt originalA.txt originalB.txt originalC.txt
