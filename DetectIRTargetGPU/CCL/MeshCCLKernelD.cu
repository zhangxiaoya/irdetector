#include "MeshCCLKernelD.cuh"
#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <core/core.hpp>
#include "../Assistants/ShowFrame.hpp"

__device__ int IntMin(int a, int b)
{
	return a < b ? a : b;
}

__global__ void MeshKernelDScanning(unsigned char* frame, int* label, int* reference, const int width, const int height, bool* iterationFlag)
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
		*iterationFlag = true;
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


__device__ int IMin(int a, int b)
{
	return a < b ? a : b;
}

__device__ unsigned char diff(unsigned char a, unsigned char b)
{
	return abs(a - b);
}

__global__ void InitCCL(int labelList[], int reference[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	labelList[id] = reference[id] = id;
}

__global__ void Scanning(unsigned char frame[], int labelList[], int reference[], bool* markFlag, int N, int width, int height, unsigned char threshold)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && diff(value, frame[id - width]) <= threshold)
		label = IMin(label, labelList[id - width]);
	if (id + width < N  && diff(value, frame[id + width]) <= threshold)
		label = IMin(label, labelList[id + width]);

	int col = id % width;

	if (col > 0 && diff(value, frame[id - 1]) <= threshold)
		label = IMin(label, labelList[id - 1]);
	if (col + 1 < width  && diff(value, frame[id + 1]) <= threshold)
		label = IMin(label, labelList[id + 1]);

	if (label < labelList[id])
	{
		reference[labelList[id]] = label;
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

	if (id - width >= 0 && diff(value, frame[id - width]) <= threshold)
		label = IMin(label, labelList[id - width]);

	if (id + width < N  && diff(value, frame[id + width]) <= threshold)
		label = IMin(label, labelList[id + width]);

	int col = id % width;
	if (col > 0)
	{
		if (diff(value, frame[id - 1]) <= threshold)
			label = IMin(label, labelList[id - 1]);
		if (id - width - 1 >= 0 && diff(value, frame[id - width - 1]) <= threshold)
			label = IMin(label, labelList[id - width - 1]);
		if (id + width - 1 < N  && diff(value, frame[id + width - 1]) <= threshold)
			label = IMin(label, labelList[id + width - 1]);
	}
	if (col + 1 < width)
	{
		if (diff(value, frame[id + 1]) <= threshold)
			label = IMin(label, labelList[id + 1]);
		if (id - width + 1 >= 0 && diff(value, frame[id - width + 1]) <= threshold)
			label = IMin(label, labelList[id - width + 1]);
		if (id + width + 1 < N  && diff(value, frame[id + width + 1]) <= threshold)
			label = IMin(label, labelList[id + width + 1]);
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


void MeshCCL(unsigned char* frame, int* labels, int width, int height)
{
	unsigned char* FrameDataOnDevice;
	int* LabelListOnDevice;
	int* ReferenceOnDevice;

	auto degreeOfConnectivity = 4;
	unsigned char threshold = 0;

	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&ReferenceOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BlockX - 1) / BlockX, (height + BlockY - 1) / BlockY);
	dim3 threads(BlockX, BlockY);

	InitCCL<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice, width, height);

	while (true)
	{
		auto markFlagOnHost = false;
		cudaMemcpy(markFlagOnDevice, &markFlagOnHost, sizeof(bool), cudaMemcpyHostToDevice);

		if (degreeOfConnectivity == 4)
			Scanning<<<grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, markFlagOnDevice, N, width, height, threshold);
		else
			scanning8<<<grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, ReferenceOnDevice, markFlagOnDevice, N, width, height, threshold);

		cudaMemcpy(&markFlagOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFlagOnHost)
		{
			analysis<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice, width, height);
			cudaThreadSynchronize();
			labeling<<<grid, threads>>>(LabelListOnDevice, ReferenceOnDevice, width, height);
		}
		else
		{
			break;
		}
	}

	cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cv::Mat labelImg;
	ShowFrame::ToMat<int>(labels, width, height, labelImg, CV_32FC1);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
	cudaFree(ReferenceOnDevice);
}