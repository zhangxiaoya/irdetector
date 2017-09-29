#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "../Checkers/CheckCUDAReturnStatus.h"

typedef unsigned short(*pointFunction_t)(unsigned short, unsigned short);

__device__ unsigned short USMinOnDevice(unsigned short a, unsigned short b)
{
	return (a < b) ? a : b;
}

__device__ unsigned short USMaxOnDevice(unsigned short a, unsigned short b)
{
	return (a > b) ? a : b;
}

__device__ inline int IMinOnDevice(int a, int b)
{
	return a > b ? b : a;
}

__device__ inline int IMaxOnDevice(int a, int b)
{
	return a > b ? a : b;
}

__device__ void FilterStep2Kernel(unsigned short* srcFrameOnDevice,
                                  unsigned short* dstFrameOnDevice,
                                  int width,
                                  int height,
                                  int tileWidth,
                                  int tileHeight,
                                  const int radius,
                                  const pointFunction_t pPointOperation)
{
	extern __shared__ unsigned short smem[];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	auto x = bx * tileWidth + tx;
	auto y = by * tileHeight + ty - radius;

	smem[ty * blockDim.x + tx] = 0;
	__syncthreads();
	if (x >= width || y < 0 || y >= height)
	{
		return;
	}
	smem[ty * blockDim.x + tx] = srcFrameOnDevice[y * width + x];
	__syncthreads();
	if (y < (by * tileHeight) || y >= ((by + 1) * tileHeight))
	{
		return;
	}
	auto smem_thread = &smem[(ty - radius) * blockDim.x + tx];
	auto val = smem_thread[0];
#pragma unroll
	for (auto yy = 1; yy <= 2 * radius; yy++)
	{
		val = pPointOperation(val, smem_thread[yy * blockDim.x]);
	}
	dstFrameOnDevice[y * width + x] = val;
}

__device__ void FilterStep1Kernel(unsigned short* srcFrameOnDevice,
                                  unsigned short* dstFrameOnDevice,
                                  int width,
                                  int height,
                                  int tileWidth,
                                  int tileHeight,
                                  const int radius,
                                  const pointFunction_t pPointOperation)
{
	extern __shared__ unsigned short smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	auto x = bx * tileWidth + tx - radius;
	auto y = by * tileHeight + ty;
	smem[ty * blockDim.x + tx] = 0;
	__syncthreads();
	if (x < 0 || x >= width || y >= height)
	{
		return;
	}
	smem[ty * blockDim.x + tx] = srcFrameOnDevice[y * width + x];
	__syncthreads();
	if (x < (bx * tileWidth) || x >= ((bx + 1) * tileWidth))
	{
		return;
	}
	auto smem_thread = &smem[ty * blockDim.x + tx - radius];
	auto val = smem_thread[0];
#pragma unroll
	for (auto xx = 1; xx <= 2 * radius; xx++)
	{
		val = pPointOperation(val, smem_thread[xx]);
	}
	dstFrameOnDevice[y * width + x] = val;
}

__global__ void DilationFilterStep1(unsigned short* srcFrameOnDevice,
                                    unsigned short* dstFrameOnDevice,
                                    int width,
                                    int height,
                                    int tileWidh,
                                    int tileHeight,
                                    const int radius)
{
	FilterStep1Kernel(srcFrameOnDevice, dstFrameOnDevice, width, height, tileWidh, tileHeight, radius, USMaxOnDevice);
}

__global__ void DilationFilterStep2(unsigned short* srcFrameOnDevice,
                                    unsigned short* dstFrameOnDevice,
                                    int width,
                                    int height,
                                    int tileWidth,
                                    int tileHeight,
                                    const int radius)
{
	FilterStep2Kernel(srcFrameOnDevice, dstFrameOnDevice, width, height, tileWidth, tileHeight, radius, USMaxOnDevice);
}

void DilationFilter(unsigned short* srcFrameOnDevice,
                    unsigned short* dstFrameOnDevice,
                    int width,
                    int height,
                    int radius)
{
	auto status = true;
	unsigned short* firstStepResultOnDevice;
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&firstStepResultOnDevice), width * height * sizeof(unsigned short)), status);

	auto tileWidthOfStep1 = 256;
	auto tileHeightOfStep1 = 1;

	dim3 blockOfStep1(tileWidthOfStep1 + (2 * radius), tileHeightOfStep1);
	dim3 gridOfStep1(ceil(static_cast<float>(width) / tileWidthOfStep1), ceil(static_cast<float>(height) / tileHeightOfStep1));

	auto tileWidthOfStep2 = 4;
	auto tileHeightOfStep2 = 64;

	dim3 blockOfStep2(tileWidthOfStep2, tileHeightOfStep2 + (2 * radius));
	dim3 gridOfStep2(ceil(static_cast<float>(width) / tileWidthOfStep2), ceil(static_cast<float>(height) / tileHeightOfStep2));

	DilationFilterStep1<<<gridOfStep1, blockOfStep1, blockOfStep1.y * blockOfStep1.x >>>(srcFrameOnDevice, firstStepResultOnDevice, width, height, tileWidthOfStep1, tileHeightOfStep1, radius);
	CheckCUDAReturnStatus(cudaDeviceSynchronize(),status);
	DilationFilterStep2<<<gridOfStep2, blockOfStep2, blockOfStep2.y * blockOfStep2.x >>>(firstStepResultOnDevice, dstFrameOnDevice, width, height, tileWidthOfStep2, tileHeightOfStep2, radius);
	CheckCUDAReturnStatus(cudaDeviceSynchronize(), status);

	CheckCUDAReturnStatus(cudaFree(firstStepResultOnDevice), status);
}

__global__ void NaiveDilationKernel(unsigned short* srcFrameOnDevice, unsigned short* dstFrameOnDevice, int width, int height, int radius)
{
	int colCount = blockIdx.x * blockDim.x + threadIdx.x;
	int rowCount = blockIdx.y * blockDim.y + threadIdx.y;

	if (rowCount >= height || colCount >= width)
	{
		return;
	}
	unsigned int startRow = IMaxOnDevice(rowCount - radius, 0);
	unsigned int endRow = IMinOnDevice(height - 1, rowCount + radius);
	unsigned int startCol = IMaxOnDevice(colCount - radius, 0);
	unsigned int endCol = IMinOnDevice(width - 1, colCount + radius);

	unsigned short maxValue = 0;

	for (int r = startRow; r <= endRow; r++)
	{
		for (int c = startCol; c <= endCol; c++)
		{
			maxValue = USMaxOnDevice(maxValue, srcFrameOnDevice[r * width + c]);
		}
	}
	dstFrameOnDevice[rowCount * width + colCount] = maxValue;
}

void NaiveDilation(unsigned short* srcFrameOnDevice, unsigned short* dstFrameOnDevice, int width, int height, int radius)
{
	auto status = true;
	dim3 block(32, 32);
	dim3 grid(ceil(static_cast<float>(width) / block.x), ceil(static_cast<float>(height) / block.y));
	NaiveDilationKernel<<<grid, block >>>(srcFrameOnDevice, dstFrameOnDevice, width, height, radius);
	CheckCUDAReturnStatus(cudaDeviceSynchronize(), status);
	auto cudaerr = cudaDeviceSynchronize();
}