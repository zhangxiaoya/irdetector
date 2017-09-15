#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include "segementationHelper.cuh"
#include "../Assistants/ShowFrame.hpp"
#include "../CCL/MeshCCLKernelD.cuh"
#include "../Models/FourLimits.h"
#include "../Models/Point.h"
#include "../Models/ObjectRect.h"
#include "../Checkers/CheckPerf.h"

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

void Segmentation(unsigned char* frameOnHost, int width, int height)
{
	int* labelsOnHost;
	int* labelsOnDevice;
	int* referenceOfLablesOnDevice;
	bool* modificationFlagOnDevice;
	unsigned char* frameOnDevice;

	cudaMallocHost((void**)&labelsOnHost, width * height * sizeof(int));
	cudaMalloc((void**)&frameOnDevice, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&labelsOnDevice, sizeof(int) * width * height);
	cudaMalloc((void**)&referenceOfLablesOnDevice, sizeof(int) * width* height);
	cudaMalloc((void**)&modificationFlagOnDevice, sizeof(bool));

	cudaMemcpy(frameOnDevice, frameOnHost, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

	cv::Mat img;
	ShowFrame::ToMat<unsigned char>(frameOnHost, width, height, img, CV_8UC1);

	ShowFrame::ToTxt<unsigned char>(frameOnHost, "data.txt", width, height);

	CheckPerf(MeshCCL(frameOnDevice, labelsOnDevice, referenceOfLablesOnDevice,modificationFlagOnDevice,width, height),"Mesh CCL");

	cudaMemcpy(labelsOnHost, labelsOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

	ShowFrame::ToTxt<int>(labelsOnHost,"lables.txt", width, height);

	auto allObjects = new FourLimits[width * height];

	CheckPerf(GetAllObjects(width, height, labelsOnHost, allObjects),"All Objects");

	auto allObjectRects = new ObjectRect[width * height];

	CheckPerf(do_work(width, height, allObjects, allObjectRects), "To Rect");

	ShowFrame::DrawRectangles(frameOnHost, allObjectRects, width, height);

	delete[] allObjectRects;
	delete[] allObjects;
	cudaFreeHost(labelsOnHost);
	cudaFree(labelsOnDevice);
	cudaFree(referenceOfLablesOnDevice);
	cudaFree(frameOnDevice);
	cudaFree(modificationFlagOnDevice);
}
