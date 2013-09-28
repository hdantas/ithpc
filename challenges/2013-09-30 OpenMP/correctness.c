#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define DIM 256
#define MAXNUM 100

void reset();
void init();
void outerloop();
void innerloop();
void seq();
void print(char *file);
void twoD();

int y[DIM];
int x[DIM];
int A[DIM][DIM];


int main() {

    init();

    printf("Sequential\n");
    reset();
    seq();
    print("seq.txt");
    
    printf("Outerloop\n");
    reset();
    outerloop();
    print("outer.txt");

    printf("Innerloop\n");
    reset();
    innerloop();
    print("inner.txt");

    printf("2D\n");
    reset();
    twoD();
    print("2D.txt");

    return 0;
}

void reset(){
    int i;
    for( i = 0; i < DIM; i++){
        y[i] = 0;
    }
}
void init() {
    int i, j;
    
    srand(time(NULL) );
    
    for(i = 0; i < DIM; i++){
            x[i] = rand() % MAXNUM;
    }

    for(i = 0; i < DIM; i++){
        for(j = 0; j < DIM; j++){
            A[i][j] = rand() % MAXNUM;
        }
    }   
}
void outerloop() {
    int i, j;

    #pragma omp parallel for private(j)
    for(i = 0; i < DIM; i++) {
        y[i] = 0; 
        for(j = 0; j < DIM; j++) {
            y[i] = y[i] + A[i][j] * x[j]; 
        }
    }

}

void innerloop() {
    int i, j;
    int tmp;
    
    for(i = 0; i < DIM; i++) {
        tmp = 0;
        #pragma omp parallel for reduction(+ : tmp)
        for(j = 0; j < DIM; j++) {
            tmp = tmp + A[i][j] * x[j];
        }
        y[i] = tmp;
    }
}

void seq() {
    int i, j;
    for(i = 0; i < DIM; i++) {
        y[i] = 0; 
        for(j = 0; j < DIM; j++) {
            y[i] = y[i] + A[i][j] * x[j];
        }
    }
}

void print(char *txt) {
    FILE *f = fopen(txt, "w");
        
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    int i;
    for(i = 0; i < DIM; i++) {
        fprintf(f, "\ty[%d] = %d\n", i, y[i]);
    }    

    fclose(f);
}

void twoD() { 
    int i, j, k, out;
    int tmp[DIM * DIM];

    #pragma omp parallel for private(i, j)
    for (k = 0; k < DIM * DIM; k++) {
        i = (int) k / DIM;
        j = k % DIM;
        tmp[k] = A[i][j] * x[j];
    }

    for(i = 0; i < DIM; i++) {
        out = 0;
        #pragma omp parallel for reduction(+ : out)
        for(j = 0; j < DIM; j++) {
            out = out + tmp[i * DIM + j];
        }
        y[i] = out;
    }
}