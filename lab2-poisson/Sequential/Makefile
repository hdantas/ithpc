all: run

compile: SEQ_Poisson.c
	gcc -o $@.out $^

run: compile
	./$^.out

clean:
	rm -rfv *.o *.out output*.dat
