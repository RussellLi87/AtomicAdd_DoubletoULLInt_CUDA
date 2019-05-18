
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>

/*********************************************************************************************************************
				DECLARATION OF MACROS AND GLOBAL VARIABLES
*********************************************************************************************************************/
#define N 512
#define ITERATIONS 10

double average, variance, std_dev, sum1, sum2, minV, maxV, minG, maxG;
int	   DECIMALPLACE = 1;

__device__ double pow(int DECIMALPLACE) {
	double temp = 10.0;
	for (int i = DECIMALPLACE -1 ; i > 0; i--)
	{
		temp *= 10;
	}
	return temp;
}

__global__ void AtomicAddKernelULLInt(double *g_inData, unsigned long long int *g_outData, int DECIMALPLACE) {
	extern __shared__ double sharedMem[];

	int threadID = threadIdx.x;
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;

	sharedMem[threadID] = g_inData[globalID];
	__syncthreads();
	
	
	unsigned long long int temp = round(sharedMem[threadID] * pow(DECIMALPLACE));
	atomicAdd(&g_outData[blockIdx.x], temp);
	__syncthreads();
}

__global__ void AddAbsMinKernel(double* g_inData, double min) {
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;

	g_inData[globalID] += abs(min);
}


/***
Random number generator in the range of [0,1)
RAND_MAX = 32767
*/
double GetRand()
{
	//return (double)rand() / (double)RAND_MAX - 0.5f;

	//return (double)rand() / (double)RAND_MAX;

	return (double)rand() / (double)RAND_MAX * 10000000000000000;
}

/****
	Find maximum, minimum, mean, variance and standard deviation of a vector
**/
void Statistics(double* h_inData, int n) {
	average = 0, variance = 0, std_dev = 0, sum1 = 0, sum2 = 0, minV = 0, maxV = 0;
	minV = h_inData[0];

	/*  Compute the sum of all elements
		and the max and min	*/
	for (int i = 0; i < n; i++)
	{
		sum1 += h_inData[i];
		maxV = fmax(maxV, h_inData[i]);
		minV = fmin(minV, h_inData[i]);
	}
	average = sum1 / (double)n;

	/*  Compute  variance  and standard deviation  */
	for (int i = 0; i < n; i++)
	{
		sum2 += powf((h_inData[i] - average), 2.0);
	}
	variance = sum2 / (double)n;
	std_dev = sqrt(variance);

	printf("Maximum number is %.50f\n", maxV);
	printf("Minimum number is %.50f\n", minV);
	printf("Average of all elements is %.50f\n", average);
	printf("variance of all elements is %.10f\n", variance);
	printf("Standard deviation is %.6f\n", std_dev);
}


int FindMaxDecimalPlaces(int blockSize, int gridSize) {
	double temp;
	int places = -1;
	
	temp = LLONG_MAX / (maxG * blockSize * gridSize);
	while (temp > 1)
	{
		temp /= 10;
		places++;
	}
	std::cout << "Largest decimal places is " << places << std::endl;
	return places;

}


/*********************************************************************************************************************
					                MAIN
*********************************************************************************************************************/
int main() {
	/*********************************************************************************************************************
							CONFIGURE CUDA GRID
	*********************************************************************************************************************/
	int blockSize;
	int gridSize;
	blockSize = 512;
	gridSize = (N + blockSize - 1) / blockSize;

	/*********************************************************************************************************************
							RAW DATA GENERATION
	*********************************************************************************************************************/
	double *h_inData = (double*)malloc(N * sizeof(double));
	double *d_inData; cudaMalloc(&d_inData, N * sizeof(double));

	srand(100);

	for (int i = 0; i < N; i++)
	{
		h_inData[i] = GetRand();
	}

	double testSum = 0;
	for (int i = 0; i < N; i++)
	{
		testSum += h_inData[i];
	}
	printf("sum %.50lf\n", testSum);

	/*********************************************************************************************************************
							DATA STATISTICS
	*********************************************************************************************************************/
	std::cout << "Raw data analysis:" << std::endl;
	Statistics(h_inData, N);
	minG = minV;
	maxG = maxV;
	std::cout << "Raw data analysis ends." << std::endl;

	/*********************************************************************************************************************
							CHECK IF ANY ELEMENT IS NEGATIVE
	*********************************************************************************************************************/
	if (minG < 0) // if yes, launch a kernel to turn all values into positive
	{
		cudaMemcpy(d_inData, h_inData, N * sizeof(double), cudaMemcpyHostToDevice);
		AddAbsMinKernel << < gridSize, blockSize >> > (d_inData, minG);
		cudaMemcpy(h_inData, d_inData, N * sizeof(double), cudaMemcpyDeviceToHost);
		Statistics(h_inData, N);
	}


	/*********************************************************************************************************************
							FIND MAXIMUM DECIMAL PLACES
	*********************************************************************************************************************/
	DECIMALPLACE = FindMaxDecimalPlaces(blockSize, gridSize);
	//DECIMALPLACE = 16;

	cudaMemcpy(d_inData, h_inData, N * sizeof(double), cudaMemcpyHostToDevice);

	long long int sum;
	double* results = (double*)malloc(sizeof(double) * ITERATIONS);

	unsigned long long int *h_outData = (unsigned long long int*)malloc(gridSize * sizeof(unsigned long long int));
	for (int i = 0; i < ITERATIONS; i++)
	{
		unsigned long long int *d_outData; cudaMalloc(&d_outData, gridSize * sizeof(unsigned long long int));
		sum = 0;
		AtomicAddKernelULLInt << <gridSize, blockSize, blockSize * sizeof(double) >> > (d_inData, d_outData, DECIMALPLACE);
		cudaDeviceSynchronize();

		cudaMemcpy(h_outData, d_outData, gridSize * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

		for (int j = 0; j < gridSize; j++)
		{
			sum += h_outData[j];
		}
		
		if (minG < 0)
		{
			results[i] = ((double)sum / pow(10, DECIMALPLACE)) - ((abs(minG) * N));
		}
		else
		{
			results[i] = (double)sum / pow(10, DECIMALPLACE);
		}
		printf("%ith Sum of this vector is %.50lf.\n", i, results[i]);
	}
	Statistics(results, ITERATIONS);

	double acc = 1 - (abs(results[0] - testSum) / testSum);
	printf("Accuracy: %.50lf", acc);

}
