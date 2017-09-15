#include "MeshCCLKernelD.cuh"
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <core/core.hpp>
#include "../Assistants/ShowFrame.hpp"

__device__ int IntMinOnDevice(int a, int b)
{
	return a < b ? a : b;
}

__device__ unsigned char UCDiffOnDevice(unsigned char a, unsigned char b)
{
	return abs(a - b);
}

__global__ void InitCCLOnDevice(int labelsOnDevice[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelsOnDevice[id] = reference[id] = id;
}

__global__ void Scanning(unsigned char* frameOnDevice, int* labelsOnDevice, int* reference, bool* markFlag, int N, int width, int height, unsigned char threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	unsigned char value = frameOnDevice[id];
	int label = N;

	if (id - width >= 0 && UCDiffOnDevice(value, frameOnDevice[id - width]) <= threshold)
		label = IntMinOnDevice(label, labelsOnDevice[id - width]);
	if (id + width < N  && UCDiffOnDevice(value, frameOnDevice[id + width]) <= threshold)
		label = IntMinOnDevice(label, labelsOnDevice[id + width]);

	int col = id % width;

	if (col > 0 && UCDiffOnDevice(value, frameOnDevice[id - 1]) <= threshold)
		label = IntMinOnDevice(label, labelsOnDevice[id - 1]);
	if (col + 1 < width  && UCDiffOnDevice(value, frameOnDevice[id + 1]) <= threshold)
		label = IntMinOnDevice(label, labelsOnDevice[id + 1]);

	if (label < labelsOnDevice[id])
	{
		reference[labelsOnDevice[id]] = label;
		*markFlag = true;
	}
}

__global__ void scanning8(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = x + y * blockDim.x * gridDim.x;

	if (id >= N) return;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && UCDiffOnDevice(value, frame[id - width]) <= threshold)
		label = IntMinOnDevice(label, labelList[id - width]);

	if (id + width < N  && UCDiffOnDevice(value, frame[id + width]) <= threshold)
		label = IntMinOnDevice(label, labelList[id + width]);

	int col = id % width;
	if (col > 0)
	{
		if (UCDiffOnDevice(value, frame[id - 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id - 1]);
		if (id - width - 1 >= 0 && UCDiffOnDevice(value, frame[id - width - 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id - width - 1]);
		if (id + width - 1 < N  && UCDiffOnDevice(value, frame[id + width - 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id + width - 1]);
	}
	if (col + 1 < width)
	{
		if (UCDiffOnDevice(value, frame[id + 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id + 1]);
		if (id - width + 1 >= 0 && UCDiffOnDevice(value, frame[id - width + 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id - width + 1]);
		if (id + width + 1 < N  && UCDiffOnDevice(value, frame[id + width + 1]) <= threshold)
			label = IntMinOnDevice(label, labelList[id + width + 1]);
	}

	if (label < labelList[id])
	{
		reference[labelList[id]] = label;
		*markFlag = true;
	}
}

__global__ void analysis(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	int label = labelList[id];
	int ref;
	if (label == id)
	{
		do
		{
			ref = label;
			label = reference[ref];
		} while (ref ^ label);
		reference[id] = label;
	}
}

__global__ void labeling(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelList[id] = reference[reference[labelList[id]]];
}


void MeshCCL(unsigned char* frameOnDevice, int* labelsOnDevice, int* referenceOfLabelsOnDevice, int width, int height)
{
	auto degreeOfConnectivity = 4;
	unsigned char threshold = 0;

	auto N = width * height;

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BlockX - 1) / BlockX, (height + BlockY - 1) / BlockY);
	dim3 threads(BlockX, BlockY);

	InitCCLOnDevice<<<grid, threads>>>(labelsOnDevice, referenceOfLabelsOnDevice, width, height);

	while (true)
	{
		auto markFlagOnHost = false;
		cudaMemcpy(markFlagOnDevice, &markFlagOnHost, sizeof(bool), cudaMemcpyHostToDevice);

		if (degreeOfConnectivity == 4)
			Scanning<<<grid, threads>>>(frameOnDevice, labelsOnDevice, referenceOfLabelsOnDevice, markFlagOnDevice, N, width, height, threshold);
		else
			scanning8<<<grid, threads>>>(frameOnDevice, labelsOnDevice, referenceOfLabelsOnDevice, markFlagOnDevice, N, width, height, threshold);

		cudaMemcpy(&markFlagOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFlagOnHost)
		{
			analysis<<<grid, threads>>>(labelsOnDevice, referenceOfLabelsOnDevice, width, height);
			cudaThreadSynchronize();
			labeling<<<grid, threads>>>(labelsOnDevice, referenceOfLabelsOnDevice, width, height);
		}
		else
		{
			break;
		}
	}
}