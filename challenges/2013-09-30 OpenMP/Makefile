all: compare

template:
	gcc template.c -o template.out -Wall -fopenmp && ./template.out

openMP:
	gcc OpenMP.c -o OpenMP.out -Wall -fopenmp && ./OpenMP.out

compare:
	diff seq.txt inner.txt && diff seq.txt outer.txt && diff seq.txt 2D.txt

clean:
	rm -fv seq.txt inner.txt outer.txt 2D.txt *.out