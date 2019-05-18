#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

//#ifndef ANNCUDA_RANDOM
//#define ANNCUDA_RANDOM() ((double)rand() / RAND_MAX)
//#endif

	__global__ void AtomicAddKernelULLInt(double *g_inData, unsigned long long int *g_outData, int DECIMALPLACE);

	__global__ void AddAbsMinKernel(double* g_inData, double min);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_CUH