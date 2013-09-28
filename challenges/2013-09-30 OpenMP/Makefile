all: openMP

correctness: correctness.c
	gcc $^ -o $@.out -Wall -fopenmp && ./$@.out

openMP: OpenMP.c
	gcc $^ -o $@.out -Wall -fopenmp && ./$@.out

compare:
	diff seq.txt inner.txt && diff seq.txt outer.txt && diff seq.txt 2D.txt

clean:
	-rm -v seq.txt inner.txt outer.txt 2D.txt *.out