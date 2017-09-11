#include "MeshKernelD.cuh"
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

void MeshCCL(unsigned char* frame, int* label, int width, int height)
{
	int* reference;
	cudaMalloc(reinterpret_cast<void**>(&reference), sizeof(int) * width * height);

	dim3 block(BlockX, BlockY);
	dim3 grid((width + BlockX - 1) / BlockX, (height + BlockY - 1) / BlockY);

	InitCCL<<<grid, block>>>(label, reference, width, height);

	cudaError_t cudaStatus;
	auto iterationFlag = false;
	do
	{
		MeshKernelDScanning<<<grid, block>>>(frame, label, reference, width, height, iterationFlag);
		cudaStatus = cudaDeviceSynchronize();

		MeshKernelDAnalysis<<<grid, block >>>(label, reference, width, height);
		cudaStatus = cudaDeviceSynchronize();

		MeshKernelDLabelling<<<grid, block >>>(label, reference, width, height);
		cudaStatus = cudaDeviceSynchronize();

	}
	while (iterationFlag);

	cudaStatus = cudaFree(reference);

	auto delay = 0;
}

__device__ int IntMin(int a, int b)
{
	return a < b ? a : b;
}

__global__ void InitCCL(int* label, int* reference, int width, int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height)
		return;

	auto id = x + y * width;

	label[id] = reference[id] = id;
}

__global__ void MeshKernelDScanning(unsigned char* frame, int* label, int* reference, const int width, const int height, bool& iterationFlag)
{
//	__shared__ unsigned char smem[BlockY][BlockX];

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if(x >= width || y >= height)
		return;

	int id = x + y * width;

//	int blockX = threadIdx.x;
//	int blockY = threadIdx.y;
//	smem[blockY][blockX] = frame[id];
//	__syncthreads();

	int currentLabel = label[id];
	int newLabel = width * height;
	unsigned char val = frame[id];

	if (y > 0 && val == frame[id - gridDim.x * blockDim.x])
		newLabel = IntMin(newLabel, label[id - gridDim.x * blockDim.x]);
	if (y < height - 1 && val == frame[id + gridDim.x * blockDim.x])
		newLabel = IntMin(newLabel, label[id + gridDim.x * blockDim.x]);
	if (x > 0 && val == frame[id - 1])
		newLabel = IntMin(newLabel, label[id - 1]);
	if (x < width - 1 && val == frame[id + 1])
		newLabel = IntMin(newLabel, label[id + 1]);

	if(newLabel < currentLabel)
	{
		reference[currentLabel] = newLabel;
		iterationFlag = true;
	}
}

__global__ void MeshKernelDAnalysis(int* label, int* reference, const int width, const int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height)
		return;

	int id = x + y * gridDim.x * blockDim.x;

	auto newLabel = label[id];
	int ref;
	if (newLabel == id)
	{
		do
		{
			ref = newLabel;
			newLabel = reference[ref];
		}
		while (ref ^ newLabel);

		reference[id] = newLabel;
	}
}

__global__ void MeshKernelDLabelling(int* label, int* reference, const int width, const int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height)
		return;

	int id = x + y * gridDim.x * blockDim.x;

	auto oldlabel = label[id];
	label[id] = reference[reference[oldlabel]];
}
