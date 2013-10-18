DIM = 1000 1000
FLAGS = -lrt -lm -Wall

all: run-seq

original: matmul.c
	gcc -o $@.out $(FLAGS) $^

run-original: original
	./$^.out $(DIM)

seq: lab5.c
	gcc -o $@.out $(FLAGS) -fopenmp $^

run-seq: seq
	./$^.out $(DIM)

diffA: originalA.txt A.txt
	diff $^

diffB: originalB.txt B.txt
	diff $^

diffC: originalC.txt C.txt
	diff $^	

compare: diffA diffB diffC

clean:
	rm -rfv *~ *.out

clean-all: clean
	rm -rfv A.txt B.txt C.txt originalA.txt originalB.txt originalC.txt