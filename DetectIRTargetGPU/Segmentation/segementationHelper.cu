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

struct FourLimits
{
	FourLimits(): top(-1), bottom(-1), left(-1), right(-1)
	{
	}
	int top;
	int bottom;
	int left;
	int right;
};

void GetAllObjects(int width, int height, int* labelsOnHost, FourLimits* allObjects)
{
	// top
	for(auto r = 0; r < height;++r)
	{
		for(auto c = 0;c < width;++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].top == -1)
				allObjects[label].top = r;
		}
	}
	// bottom
	for (auto r = height -1; r >= 0; --r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].bottom == -1)
				allObjects[label].bottom = r;
		}
	}

	// left
	for (auto c = 0; c < width; ++c)
	{
		for (auto r = 0; r < height; ++r)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].left == -1)
				allObjects[label].left = c;
		}
	}
	// right
	for (auto c = width -1; c >= 0; --c)
	{
		for (auto r = 0; r < height; ++r)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].right == -1)
				allObjects[label].right = c;
		}
	}
}

void Segmentation(unsigned char* frame, int width, int height)
{
	int* labelsOnHost;
	cudaMallocHost(reinterpret_cast<void**>(&labelsOnHost), width * height * sizeof(int));

	cv::Mat img;
	ShowFrame::ToMat<unsigned char>(frame, width, height, img, CV_8UC1);

	ShowFrame::ToTxt<unsigned char>(frame, "data.txt", width, height);

	CheckPerf(MeshCCL(frame, labelsOnHost, width, height),"Mesh CCL");

	ShowFrame::ToTxt<int>(labelsOnHost,"lables.txt", width, height);

	auto allObjects = new FourLimits[WIDTH * HEIGHT];

	CheckPerf(GetAllObjects(width, height, labelsOnHost, allObjects),"All Objects");

	delete[] allObjects;
	cudaFreeHost(labelsOnHost);
}
