#include "segementationHelper.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "../CCL/MeshKernelD.cuh"
#include <iostream>
#include "../Assistants/ShowFrame.hpp"
#include <iomanip>
#include <Windows.h>

#define CHECK(call)                                                        \
{                                                                          \
	const cudaError_t error = call;                                        \
	if(error != cudaSuccess)                                               \
	{                                                                      \
		printf("Error: %s: %d,  " __FILE__, __LINE__);                     \
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
	}                                                                      \
}

#define CheckPerf(call, message)                                                                             \
{                                                                                                            \
	LARGE_INTEGER t1, t2, tc;                                                                                \
	QueryPerformanceFrequency(&tc);                                                                          \
	QueryPerformanceCounter(&t1);                                                                            \
	call;                                                                                                    \
	QueryPerformanceCounter(&t2);                                                                            \
	printf("Operation of %20s Use Time:%f\n", message, (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);       \
};

void Segmentation(unsigned char* frame, int width, int height)
{
	int* labelsOnHost;
	cudaMallocHost(reinterpret_cast<void**>(&labelsOnHost), width * height * sizeof(int));

	cv::Mat img;
	ShowFrame::ToMat<unsigned char>(frame, width, height, img, CV_8UC1);

	ShowFrame::ToTxt<unsigned char>(frame, "data.txt", width, height);

	CheckPerf(MeshCCL(frame, labelsOnHost, width, height),"Mesh CCL");

	ShowFrame::ToTxt<int>(labelsOnHost,"lables.txt", width, height);

	cudaFreeHost(labelsOnHost);
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
