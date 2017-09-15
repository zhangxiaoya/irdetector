#pragma once
#include <stdio.h>
#define CheckCUDAReturnStatus(call, status)                                 \
{                                                                           \
	const cudaError_t error = call;                                         \
	if(error != cudaSuccess)                                                \
	{																		\
		status = false;														\
		printf("Error: %s: %d,  ", __FILE__, __LINE__);                     \
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
	}                                                                       \
}