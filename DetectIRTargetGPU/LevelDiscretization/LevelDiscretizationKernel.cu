
#include "cuda_runtime.h"
#include "LevelDiscretizationKernel.cuh"
#include <device_launch_parameters.h>

__global__ void LevelDiscretizationKernel(unsigned char* frame, int width, int height, unsigned char scale)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	auto currentIdx = y * width + x;
	frame[currentIdx] = (frame[currentIdx] / scale) * scale;
}

void LevelDiscretizationOnGPU(unsigned char* image, int width, int height, int discretizationScale)
{
	dim3 block(32, 32);
	dim3 grid((width + 31 / 32), (height + 31) / 32);

	LevelDiscretizationKernel<<<grid, block>>>(image, width, height, static_cast<unsigned char>(discretizationScale));

	auto cudaerr = cudaDeviceSynchronize();
}
