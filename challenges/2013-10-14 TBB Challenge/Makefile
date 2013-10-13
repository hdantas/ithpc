all: run-par

par: tbb.cpp
	g++ -o $@.out $^ -ltbb -std=c++0x -Wall

run-par: par
	./$^.out

compare: seq.txt par.txt
	diff $^

clean:
	rm -rfv *.out *~ *.o

