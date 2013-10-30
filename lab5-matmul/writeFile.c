#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void shiftLeft(int dim, char matrix[dim][dim], int row, int numberOfTimes);
void shiftUp (int dim, char matrix[dim][dim], int row, int numberOfTimes);

int main (int argc, char** argv){
	if (argc != 2) {
		printf("Please provide number of processors! Syntax: writeFile <numberOfProcessors>\n");
		exit(1);
	}

	int numberOfProcessors = atoi(argv[1]);
	int i, j, k, steps, size;
    int increment;
    
    steps = sqrt(numberOfProcessors);

    char orderA[steps][steps];
    char orderB[steps][steps];
    char orderC[steps][steps][2 * steps];

    // 1. Partition these matrices in square blocks p, where p is the number of processes available.
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            orderA[i][j] = 'A' + i * steps + j;
            orderB[i][j] = 'A' + i * steps + j;
        }
    }
    
    // 2. Create a matrix of processes of size p 1/2 x p 1/2 so that each process can maintain a block of A matrix and a block of B matrix.
    //init MPI

    // initial alignment
    for (i = 0; i < steps; i++) {
        shiftLeft(steps, orderA, i, i);
        shiftUp(steps, orderB, i, i);
    }


    // 5. Repeat steps 3 y 4 sqrt(p) times.
    for (k = 0; k < steps; k++) {
        for (i = 0; i < steps; i++) {
            for (j = 0; j < steps; j++) {
                orderC[i][j][2 * k + 0] = orderA[i][j];
                orderC[i][j][2 * k + 1] = orderB[i][j];
                // printf("orderC[%d][%d] = %c * %c\n",i, j, orderC[i][j][2 * k + 0], orderC[i][j][2 * k + 1]);
            }
        }

        for (i = 0; i < steps && k < steps - 1; i++) { //does not need to run in the last iteration of the for-k loop
            shiftLeft(steps, orderA, i, 1);
            shiftUp(steps, orderB, i, 1);
        }
    }

	char filename[80];
	strcpy(filename, argv[1]);
	strcat(filename,".txt");
    FILE *f = fopen(filename, "w");    
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    
    for (i = 0; i < steps; i++) {
        for (j = 0; j < steps; j++) {
            for (k = 0; k < steps; k++) {
                fprintf(f, "%c * %c", orderC[i][j][2 * k], orderC[i][j][2 * k + 1]);
                if (k < steps - 1)
                	fprintf(f, " + ");
            }
            fprintf(f, "\n");
        }
    }
    printf("Finished writing to %s\n",filename);
    fclose(f);
}


void shiftLeft(int dim, char matrix[dim][dim], int row, int numberOfTimes) {
    int j, k;
    char tmp;
    for(k = 0; k < numberOfTimes; k++) {
        tmp = matrix[row][0];
        for(j = 1; j < dim; j++) {
            matrix[row][j - 1] = matrix[row][j];
        }
        matrix[row][dim - 1] = tmp;
    }
}


void shiftUp(int dim, char matrix[dim][dim], int column, int numberOfTimes) {
    int i, k;
    char tmp;
    for(k = 0; k < numberOfTimes; k++) {
        tmp = matrix[0][column];
        for(i = 1; i < dim; i++) {
            matrix[i - 1][column] = matrix[i][column];
        }
        matrix[dim - 1][column] = tmp;
    }
}


