DIM = 500 500
FLAGS = -lrt -lm -Wall

all: run

original: matmul.c
	gcc -o $@.out $(FLAGS) $^

run-original: original
	./$^.out $(DIM)

compile: lab5.c
	gcc -o $@.out $(FLAGS) -fopenmp $^

run: compile
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

remote:
	sshpass -p 'u&Kqy43C' scp lab5.c in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	sshpass -p 'u&Kqy43C' scp Makefile in404924@fs3.das3.tudelft.nl:~/repos/ithpc/lab5-matmul
	sshpass -p 'u&Kqy43C' ssh in404924@fs3.das3.tudelft.nl "cd repos/ithpc/lab5-matmul && make"