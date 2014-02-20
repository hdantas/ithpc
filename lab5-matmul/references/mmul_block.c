#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define MAXTHREADS 4
#define BLOCK_SIZE 2
#define N 2
#define HA N * BLOCK_SIZE // height matrix A
#define WA N * BLOCK_SIZE // width matrix A
#define HB N * BLOCK_SIZE // height matrix B
#define WB N * BLOCK_SIZE // width matrix B
#define HC N * BLOCK_SIZE // height matrix C
#define WC N * BLOCK_SIZE // width matrix C

void runTest(int argc, char** argv);
void randomInit(double* matrix, int size);
void init_mat(double *A, double *B, double *C, int x_max, int y_max);
double timer();

// store the sub-matrix A
double As[BLOCK_SIZE][BLOCK_SIZE];
// store the sub-matrix B
double Bs[BLOCK_SIZE][BLOCK_SIZE];

// Matrix multiplication on the device C = A * B
void matrixMul_block_par (double* C, double* A, double* B,
	const int hA, int wA, int wB,
	const int hA_grid, const int wA_grid, const int wB_grid,
	const int nthreads)
{
	int bx, by; // Block index
	int tx, ty;
	int aBegin, aEnd, aStep, bBegin, bStep;
	int a, b, c, k;
	double Asub, Bsub, Csub;

	unsigned long int i;
	unsigned long int size = hA * wB; //size of matrix C
	memset(C, 0, size * sizeof(double));
	
	for (by = 0; by < hA_grid; by++) {
		for (bx = 0; bx < wB_grid; bx++) {
			// Index of the first sub-matrix of A processed by the block
			aBegin = wA * BLOCK_SIZE * by;
			// Index of the last sub-matrix of A processed by the block
			aEnd = aBegin + wA - 1;
			// Step size used to iterate through the sub-matrices of A
			aStep = BLOCK_SIZE;
			// Index of the first sub-matrix of B processed by the block
			bBegin = BLOCK_SIZE * bx;
			// Step size used to iterate through the sub-matrices of B
			bStep = BLOCK_SIZE * wB;

			// Loop over all the sub-matrices of A and B
			// required to compute the block sub-matrix
			for (a = aBegin, b = bBegin;
				 a < aEnd;
				 a += aStep, b += bStep) {
			// Load the matrices from main memory
				#pragma omp parallel for default(none) num_threads(nthreads) \
					shared(A, B, As, Bs, a, b, wA, wB) private(ty, tx) \
					schedule(static)
						for (ty = 0; ty < BLOCK_SIZE; ty++) {
							for (tx = 0; tx < BLOCK_SIZE; tx++) {
								// printf("As[%d][%d] = A[%d] = %lf\n", ty, tx, a + wA * ty + tx, A[a + wA * ty + tx]);
								// printf("Bs[%d][%d] = B[%d] = %lf\n", ty, tx, b + wB * ty + tx, B[b + wB * ty + tx]);
								As[ty][tx] = A[a + wA * ty + tx];
								Bs[ty][tx] = B[b + wB * ty + tx];
							}// for tx
						}// for ty

				// Multiply the two matrices togethers
				#pragma omp parallel for default(none) num_threads(nthreads) \
					shared(As, Bs, C, bx, by, wB) private(ty, tx, k, c, Asub, Bsub, Csub) \
					schedule(static)
						for (ty = 0; ty < BLOCK_SIZE; ty++) {
							for (tx = 0; tx < BLOCK_SIZE; tx++) {
								Csub = 0.0;
								for (k = 0; k < BLOCK_SIZE; ++k) {
									Asub = As[ty][k ];
									Bsub = Bs[k ][tx];
									Csub += Asub * Bsub;
								}
								c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
								C[c + wB * ty + tx] += (double) Csub;
							}// for tx
						}// for ty
			}// for each submatrix A and B
		
		}// for bx
	}// for by

}

int main(int argc, char** argv)
{
	runTest(argc, argv);
	return 0;
}

void runTest(int argc, char** argv)
{
	int nthreads = 4;
	unsigned int total_size = 0;

	double t_start, t_end, t_delta;

	// allocate host memory for matrices A and B
	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(double) * size_A;
	double* h_A = (double*) malloc(mem_size_A);
	assert(h_A);
	total_size += mem_size_A;

	unsigned int size_B = WB * HB;
	unsigned int mem_size_B = sizeof(double) * size_B;
	double* h_B = (double*) malloc(mem_size_B);
	assert(h_B);
	total_size += mem_size_B;

	unsigned int size_C = WC * HC;
	unsigned int mem_size_C = sizeof(double) * size_C;
	double* h_C = (double*) malloc(mem_size_C);
	assert(h_C);
	total_size += mem_size_C;

	// Initialize host memory
	// randomInit(h_A, size_A);
	// randomInit(h_B, size_B);
	init_mat(h_A, h_B, h_C, WA, HA);

	int hA_grid = HA / BLOCK_SIZE;
	int wA_grid = WA / BLOCK_SIZE;
	int wB_grid = WB / BLOCK_SIZE;

	int i, j;
	printf("size(A) = (%d, %d)\n", HA, WA);
	printf("size(B) = (%d, %d)\n", HB, WB);
	printf("total memory size = %6.4f (MB)\n", total_size/1048576.0);
	
	for (nthreads = 1; nthreads <= MAXTHREADS; nthreads *= 2) {
		t_start = timer();
		printf("hA_grid = %d, wA_grid = %d, wB_grid = %d\n", hA_grid, wA_grid, wB_grid);
		matrixMul_block_par(h_C, h_A, h_B, HA, WA, WB, hA_grid, wA_grid, wB_grid, nthreads);
		t_end = timer();
		t_delta = t_end - t_start;
		printf("\tthreads = %d, matrixMul cost = %lf (s)\n", nthreads, t_delta * 1.0e-9);
	}
	// printf("A = [\n");
	// for(i = 0; i < HA; i++) {
	// 	printf("\t");
	// 	for(j = 0; j < WA; j++) {
	// 		printf("%2.4lf, ", h_A[i * WA + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("    ];\n");

	// printf("B = [\n");
	// for(i = 0; i < HB; i++) {
	// 	printf("\t");
	// 	for(j = 0; j < WB; j++) {
	// 		printf("%2.4lf, ", h_B[i * WB + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("    ];\n");

	// printf("C = [\n");
	// for(i = 0; i < HC; i++) {
	// 	printf("\t");
	// 	for(j = 0; j < WC; j++) {
	// 		printf("%2.4lf, ", h_C[i * WC + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("    ];\n");
	
	for(i = 0; i < HC; i++) {
		for(j = 0; j < WC; j++) {
			printf("C.txt[%d] = %lf\n", i * WC + j, h_C[i * WC + j]);
		}
	}

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
}

void randomInit(double* matrix, int size)
{
	int i;
	int dim = sqrt(size);
	for (i = 0; i < size; i++){
		matrix[i] = (double) (i % dim);
	}
}

void init_mat(double *A, double *B, double *C, int x_max, int y_max)
{
    int i, j;
    for(i = 0; i < y_max; i++) {
        for(j = 0; j < x_max; j++) {
            A[i * x_max + j] = 0.1 * ((double)i * 0.1 +  (double)j * 0.01);
            B[i * x_max + j] = (double)i * 0.01 + (double)j * 0.1;
            
            // if (i < y_max / sqrt(numtasks) && j < x_max / sqrt(numtasks))
                C[i * x_max + j] = 0.0;
        }
    }
}

double timer()
{
    long t_val = 0; 
    struct timespec ts;
    
    clock_gettime(CLOCK_REALTIME,&ts);
    t_val = ts.tv_sec * 1e+9 + ts.tv_nsec;
        
    return (double)t_val;   
}