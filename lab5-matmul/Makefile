DIM = 256 256
P = 4
FLAGS = -lrt -lm -Wall 
OPTIMIZATION = -O2 -ffast-math -msse -msse2 -msse3 -mmmx -m3dnow -funroll-loops

#-fmodulo-sched-allow-regmoves -fgcse-lm -fgcse-sm -fgcse-las -fgcse-after-reload -fsched-spec-load-dangerous -fsched2-use-traces -fsee -fipa-struct-reorg -fipa-cp -fipa-matrix-reorg -ftree-loop-linear -ftree-parallelize-loops=4 -ftracer -funroll-loops -fvariable-expansion-in-unroller -freorder-blocks-and-partition -fwhole-program -fprofile-use -ffast-math -fno-math-errno -funsafe-math-optimizations -fassociative-math -fno-signed-zeros -fno-trapping-math -freciprocal-math -ffinite-math-only -fno-signaling-nans -fno-trapping-math -fsingle-precision-constant -fbranch-probabilities -fprofile-values -fvpt -ftracer -funroll-loops -fpeel-loops -fdata-sections -fbranch-target-load-optimize -fbranch-target-load-optimize2 -fsection-anchors 
   
	# After running a program compiled with -fprofile-arcs (see Options for Debugging Your Program or gcc), you can compile it a second time using -fbranch-probabilities, to improve optimizations based on the number of times each branch was taken. When the program compiled with -fprofile-arcs exits it saves arc execution counts to a file called sourcename.gcda for each source file. The information in this data file is very dependent on the structure of the generated code, so you must use the same source code and the same optimization options for both compilations   

all: run-par 

original: matmul.c
	gcc $^ -o $@.out $(FLAGS)

run-original: original
	./$^.out $(DIM)

omp: lab5omp.c
	gcc $^ -o $@.out $(FLAGS) -fopenmp

run-omp: omp
	./$^.out $(DIM)

par: lab5par.c
	mpicc $^ -o $@.out $(FLAGS) -fopenmp

run-par: par
	prun -v -pbs-script /usr/local/sitedep/reserve.sge/sge-script `pwd`/$^.out $(P) $(DIM)

mpi: lab5mpi.c
	mpicc $^ -o $@.out $(FLAGS) $(OPTIMIZATION) #-fopenmp

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
	scp lab5omp.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
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
	rm -rfv A.txt B.txt C.txt originalA.txt originalB.txt originalC.tx
