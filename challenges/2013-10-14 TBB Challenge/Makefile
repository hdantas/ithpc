all: run-par

par: tbb.cpp
	g++ -o $@.out $^ -ltbb -std=c++0x -Wall

run-par: par
	./$^.out

clean:
	rm -rfv *.out *~ *.o

