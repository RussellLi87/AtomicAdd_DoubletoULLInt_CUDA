#include "main.h"

__device__ double pow(int DECIMALPLACE) {
	double temp = 10.0;
	for (int i = DECIMALPLACE - 1; i > 0; i--)
	{
		temp *= 10;
	}
	return temp;
}

double GetRand()
{
	return (double)rand() / (double)RAND_MAX;
}

