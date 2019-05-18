#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include "kernels.cuh"

#define N 512
#define ITERATIONS 10


#ifdef __cplusplus
extern "C" {
#endif

	//#ifndef ANNCUDA_RANDOM
	//#define ANNCUDA_RANDOM() ((double)rand() / RAND_MAX)
	//#endif

	__device__ double pow(int DECIMALPLACE);

	/***
		Random number generator in the range of [0,1)
		RAND_MAX = 32767
	*/
	double GetRand();

	int FindMaxDecimalPlaces(int blockSize, int gridSize);
	

#ifdef __cplusplus
}
#endif

#endif // MAIN_H
