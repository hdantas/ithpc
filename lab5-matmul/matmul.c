#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <math.h>
//#define MIC
typedef double DT;

void init_mat(double *  A, double *  B, double *  C, int x_max, int y_max)
{
	int i,j;
	for(j=0; j<y_max; j++)
	{
		for(i=0; i<x_max; i++)
		{
			A[i+j*x_max] = (DT)j*0.1 + (DT)i*0.01;
			B[i+j*x_max] = (DT)j*0.01 + (DT)i*0.1;
			C[i+j*x_max] = 0.0;
		}
	}
	return ;
}

/**
	timer
**/
double timer()
{
	long t_val = 0;	
	struct timespec ts;
	
	clock_gettime(CLOCK_REALTIME,&ts);
	t_val = ts.tv_sec * 1e+9 + ts.tv_nsec;
		
	return (double)t_val;	
}


/**
	main entry
**/
int main (int argc, char** argv)
{
	int i,j,k,r;
	
	DT *  A;
	DT *  B;
	DT *  C;

	int nParamArgsCount = 0;
	char *  *  rgArgs = ((char *  * )malloc((argc*sizeof (char * ))));
	int ii;
	for (ii=1; ii<argc;  ++ ii)
	{
		if ((( * argv[ii])!='-'))
		{
			rgArgs[nParamArgsCount]=argv[ii];
			 ++ nParamArgsCount;
		}
	}
		if ((nParamArgsCount!=2))
	{
		printf("Wrong number of parameters. Syntax:\n%s <height> <width>\n", argv[0]);
		exit(-1);
	}
	int x_max = atoi(rgArgs[0]);
	int y_max = atoi(rgArgs[1]);	
	if(x_max!=y_max){
		printf("we consider square matrix only!\n");
		exit(-1);
	}
	int arr_bytes = x_max * y_max * sizeof(DT);
	int s = x_max;
	int length = s * s;
	double t_start, t_end, t_delta;
	double alpha, beta;
	t_delta = 0.0, alpha = 0.1, beta = 0.5;
	double bytes = 0;
	double flops = 0;
	
	free(rgArgs);
		
	// allocate matrix
	A=(double * )malloc(arr_bytes);
	B=(double * )malloc(arr_bytes);
	C=(double * )malloc(arr_bytes);

	init_mat(A, B, C, x_max, y_max);		

	// run matrix multiplication
	t_start = timer();
	for(j=0; j<y_max; j++) // row_
	{
   	   for(i=0; i<x_max; i++) // col_
   	   {
   	        double c = 0.0f;
   	        for(k=0; k<x_max; k++)
   	        {
   	        	double a = A[k+j*x_max];
   	        	double b = B[i+k*x_max];
   	        	c += a * b;	
   	        }
   	        C[i+j*x_max] = beta * C[i+j*x_max] + alpha * c;
		}
	}
	t_end = timer();
	t_delta = t_end - t_start;
	
	// statistics
	bytes = (double)x_max * (double)y_max * (double)4 * (double)sizeof(DT);
	flops = (double)x_max * (double)y_max * (double)x_max * 2;
	printf("time elapsed: %lf\n", t_delta*1.0e-9); 
	printf("gflops: %lf\t\n", flops/t_delta);
	printf("bandwidth: %lf\t\n", bytes/t_delta);


	// free memory space
	free(A);
	free(B);
	free(C);

	return 0;
}


