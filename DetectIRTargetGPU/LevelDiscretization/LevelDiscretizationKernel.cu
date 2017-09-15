#include "cuda_runtime.h"
#include "LevelDiscretizationKernel.cuh"
#include <device_launch_parameters.h>
#include "../Checkers/CheckCUDAReturnStatus.h"

__global__ void LevelDiscretizationKernel(unsigned char* frameOnDevice, int width, int height, unsigned char scale)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= width || y >= height)
		return;

	auto currentIdx = y * width + x;
	frameOnDevice[currentIdx] = (frameOnDevice[currentIdx] / scale) * scale;
}

void LevelDiscretizationOnGPU(unsigned char* frameOnDevice, int width, int height, int discretizationScale)
{
	auto status = true;
	dim3 block(32, 32);
	dim3 grid((width + 31) / 32, (height + 31) / 32);

	LevelDiscretizationKernel<<<grid, block>>>(frameOnDevice, width, height, static_cast<unsigned char>(discretizationScale));

	CheckCUDAReturnStatus(cudaDeviceSynchronize(), status);
}
