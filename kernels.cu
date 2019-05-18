#include "kernels.cuh"
#include "main.h"

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