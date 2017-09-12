#include "segementationHelper.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "../CCL/MeshKernelD.cuh"
#include <iostream>
#include "../Assistants/ShowFrame.hpp"

#define CHECK(call)                                                        \
{                                                                          \
	const cudaError_t error = call;                                        \
	if(error != cudaSuccess)                                               \
	{                                                                      \
		printf("Error: %s: %d,  " __FILE__, __LINE__);                     \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
	}                                                                      \
}
void Segmentation(unsigned char* frame, int width, int height)
{
	const auto bin = 15;
	unsigned char* originalFrameOnDevice;
	unsigned char* leveledFrameOnDevice;
	unsigned char* leveledFrameOnHost;
	int* labelsOnDevice;
	int* labelsOnHost;

	auto levelCount = floor(256 / 15);

	cudaMalloc(reinterpret_cast<void**>(&originalFrameOnDevice), width* height);
	cudaMalloc(reinterpret_cast<void**>(&leveledFrameOnDevice), width * height * levelCount);
	cudaMallocHost(reinterpret_cast<void**>(&leveledFrameOnHost), width * height * levelCount);

	cudaMalloc(reinterpret_cast<void**>(&labelsOnDevice), width * height * sizeof(int) * levelCount);
	cudaMallocHost(reinterpret_cast<void**>(& labelsOnHost), width * height * sizeof(int) * levelCount);

	cudaMemcpy(originalFrameOnDevice, frame, width * height, cudaMemcpyHostToDevice);

	dim3 blcok(32, 8);
	dim3 grid((width + 31) / 32, (height + 7) / 8);

	for (auto i = 0; i < levelCount; ++i)
	{
//		SplitByLevel<<<grid, blcok>>>(originalFrameOnDevice, leveledFrameOnDevice + i * width * height, width, height, bin * i);
//		auto cudaStatus = cudaDeviceSynchronize();
//		CHECK(cudaStatus);
		for(auto j = 0; j<width * height;++j)
		{
			if (frame[j] == static_cast<unsigned char>(bin*i))
				leveledFrameOnHost[i * width * height + j] = 255;
			else
				leveledFrameOnHost[i * width * height + j] = 0;
		}
	}
	for(auto i = 0; i< levelCount;++i)
	{
		std::cout << "Level " << i<<std::endl;
		ShowFrame::Show("Level", leveledFrameOnHost + i * width * height, width, height);
	}

	for (auto i = 0; i<levelCount; ++i)
	{
		MeshCCL(leveledFrameOnDevice + i * width * height, labelsOnDevice + i * width* height, width, height);
	}
	auto cudaStatus = cudaDeviceSynchronize();

	cudaMemcpy(labelsOnHost, labelsOnDevice, width * height * sizeof(int), cudaMemcpyDeviceToHost);

	for(auto i =0;i< levelCount;++i)
	{
		auto maxLabel = 0;
		for(auto j = 0;j< width * height; ++j)
		{
			if (labelsOnHost[i * width + height + j] > maxLabel)
				maxLabel = labelsOnHost[i * width + height + j];
		}
		std::cout << "Max label = " << maxLabel << std::endl;
	}

	cudaFreeHost(labelsOnHost);
	cudaFreeHost(leveledFrameOnHost);
	cudaFree(originalFrameOnDevice);
	cudaFree(leveledFrameOnDevice);
	cudaFree(labelsOnDevice);
}

struct FourLimits
{
	int top;
	int bottopm;
	int left;
	int right;
};

__global__ void GetAllFourLimits(unsigned char* frame, int width, int height)
{

}

__device__ unsigned char diff(int a,int b)
{
	return  abs((a & 0xff) - (b & 0xff));
}

__global__ void SplitByLevel(unsigned char* frame, unsigned char* dstFrame, int width, int height, unsigned char levelVal)
{
	const int x = threadIdx.x + blockDim.x + blockIdx.x;
	const int y = threadIdx.y + blockDim.y + blockIdx.y;

	if(x >= width || y >= height)
		return;
	const int id = x + y * blockDim.x * gridDim.x;


	if (diff(levelVal, frame[id]) <= 0)
		dstFrame[id] = 255;
	else
		dstFrame[id] = 0;
}
