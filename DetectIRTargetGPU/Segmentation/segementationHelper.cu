#include "segementationHelper.cuh"
#include <cuda_runtime_api.h>
#include "../CCL/MeshKernelD.cuh"
#include <iostream>
#include "../Assistants/ShowFrame.hpp"
#include <iomanip>
#include <Windows.h>

#include "../Models/FourLimits.h"
#include "../Models/Point.h"
#include "../Models/ObjectRect.h"

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

void do_work(int width, int height, FourLimits* allObjects, ObjectRect* allObjectRects)
{
	for(auto i = 0;i < width * height;++i)
	{
		if (allObjects[i].top == -1)
			continue;
		allObjectRects[i].width = allObjects[i].right - allObjects[i].left + 1;
		allObjectRects[i].height = allObjects[i].bottom - allObjects[i].top + 1;
		allObjectRects[i].lt = Point(allObjects[i].left, allObjects[i].top);
		allObjectRects[i].rb = Point(allObjects[i].right, allObjects[i].bottom);
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

	auto allObjectRects = new ObjectRect[WIDTH * HEIGHT];

	CheckPerf(do_work(width, height, allObjects, allObjectRects), "To Rect");

	delete[] allObjectRects;
	delete[] allObjects;
	cudaFreeHost(labelsOnHost);
}
