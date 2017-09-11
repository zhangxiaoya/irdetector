#include "segementationHelper.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "../CCL/MeshKernelD.cuh"

void Segmentation(unsigned char* frame, int width, int height)
{
	const auto bin = 15;
	unsigned char* originalFrameOnDevice;
	unsigned char* leveledFrameOnDevice;
	int* labelsOnDevice;

	auto levelCount = floor(256 / 15);

	cudaMalloc(reinterpret_cast<void**>(&originalFrameOnDevice), width* height);
	cudaMalloc(reinterpret_cast<void**>(&leveledFrameOnDevice), width * height * levelCount);
	cudaMalloc(reinterpret_cast<void**>(&labelsOnDevice), width * height * sizeof(int) * levelCount);

	cudaMemcpy(originalFrameOnDevice, frame, width * height, cudaMemcpyHostToDevice);

	dim3 blcok(32, 8);
	dim3 grid((width + 31) / 32, (height + 7) / 8);

	for (auto i = 0; i < levelCount; ++i)
	{
		SplitByLevel<<<grid, blcok>>>(originalFrameOnDevice, leveledFrameOnDevice + i * width * height, width, height, bin * i);
	}
	auto cudaStatus = cudaDeviceSynchronize();

	for (auto i = 0; i<levelCount; ++i)
	{
		MeshCCL(originalFrameOnDevice, labelsOnDevice + i * width* height, width, height);
	}
	cudaStatus = cudaDeviceSynchronize();
}

__global__ void SplitByLevel(unsigned char* frame, unsigned char* dstFrame, int width, int height, unsigned char levelVal)
{
	const int x = threadIdx.x + blockDim.x + blockIdx.x;
	const int y = threadIdx.y + blockDim.y + blockIdx.y;

	if(x >= width || y >= height)
		return;
	const int id = x + y * blockDim.x * gridDim.x;

	if (frame[id] == levelVal)
		dstFrame[id] = 1;
	else
		dstFrame[id] = 0;
}
