#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <cuda_runtime_api.h>
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../Headers/GlobalMainHeaders.h"
#include "../Dilations/DilatetionKernel.cuh"
#include "../LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "../CCL/MeshCCLKernelD.cuh"
#include "../Models/FourLimits.h"
#include "../Models/Point.h"
#include "../Models/ObjectRect.h"
#include "../Assistants/ShowFrame.hpp"
#include "../Common/Util.h"

class Detector
{
public:
	explicit Detector(int _width, int _height);

	~Detector();

	bool InitSpace();

	void DetectTargets(unsigned char* frame);

private:
	void CopyFrameData(unsigned char* frame);

	static void GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height);

	static void ConvertFourLimitsToRect(FourLimits* allObjects, ObjectRect* allObjectRects, int width, int height);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	void MergeObjects() const;
	void RemoveObjectWithLowContrast() const;

protected:
	bool ReleaseSpace();

private:
	int width;
	int height;

	int radius;
	int discretizationScale;

	bool isInitSpaceReady;
	bool isFrameDataReady;

	unsigned char* originalFrameOnHost;
	unsigned char* originalFrameOnDevice;
	unsigned char* dilationResultOnDevice;
	unsigned char* discretizationResultOnHost;

	int* labelsOnHost;
	int* labelsOnDevice;
	int* referenceOfLabelsOnDevice;

	bool* modificationFlagOnDevice;

	FourLimits* allObjects;
	ObjectRect* allObjectRects;

	int TARGET_WIDTH_MAX_LIMIT;
	int TARGET_HEIGHT_MAX_LIMIT;
};

inline Detector::Detector(int _width = 320, int _height = 256)
	: width(_width),
	  height(_height),
	  radius(1),
	  discretizationScale(15),
	  isInitSpaceReady(true),
	  isFrameDataReady(true),
	  originalFrameOnHost(nullptr),
	  originalFrameOnDevice(nullptr),
	  dilationResultOnDevice(nullptr),
	  discretizationResultOnHost(nullptr),
	  labelsOnHost(nullptr),
	  labelsOnDevice(nullptr),
	  referenceOfLabelsOnDevice(nullptr),
	  modificationFlagOnDevice(nullptr),
	  allObjects(nullptr),
	  allObjectRects(nullptr),
	  TARGET_WIDTH_MAX_LIMIT(20),
	  TARGET_HEIGHT_MAX_LIMIT(20)
{
}

inline Detector::~Detector()
{
	ReleaseSpace();
}

inline bool Detector::ReleaseSpace()
{
	auto status = true;
	if (this->originalFrameOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->originalFrameOnHost), status);
		if (status == true)
		{
			this->originalFrameOnHost = nullptr;
		}
	}
	if(this->labelsOnHost != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->labelsOnHost), status);
		if(status == true)
		{
			this->labelsOnHost = nullptr;
		}
	}
	if (this -> discretizationResultOnHost != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->discretizationResultOnHost), status);
		if(status == true)
		{
			this->discretizationResultOnHost = nullptr;
		}
	}
	if (this->originalFrameOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->originalFrameOnDevice), status);
		if (status == true)
		{
			this->originalFrameOnDevice = nullptr;
		}
	}
	if (this->dilationResultOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->dilationResultOnDevice), status);
		if (status == true)
		{
			this->dilationResultOnDevice == nullptr;
		}
	}
	if(this->labelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->labelsOnDevice), status);
		if(status == true)
		{
			this->labelsOnDevice = nullptr;
		}
	}
	if(this->referenceOfLabelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->referenceOfLabelsOnDevice), status);
		if(status == true)
		{
			this->referenceOfLabelsOnDevice = nullptr;
		}
	}
	if(this->modificationFlagOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->modificationFlagOnDevice), status);
		if (status == true)
		{
			this->modificationFlagOnDevice = nullptr;
		}
	}

	if(this->allObjects != nullptr)
	{
		delete[] allObjects;
	}
	if(this->allObjectRects != nullptr)
	{
		delete[] allObjectRects;
	}

	if(status == true)
	{
		logPrinter.PrintLogs("Release space success!", LogLevel::Info);
	}
	else
	{
		logPrinter.PrintLogs("Release space failed!", LogLevel::Error);
	}
	return status;
}

inline bool Detector::InitSpace()
{
	logPrinter.PrintLogs("Release space before re-init space ...", Info);
	if(ReleaseSpace() == false)
		return false;

	isInitSpaceReady = true;
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->originalFrameOnHost), sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->labelsOnHost), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->discretizationResultOnHost), sizeof(unsigned char) * width * height), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->originalFrameOnDevice), sizeof(unsigned char) * width * height),isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->dilationResultOnDevice), sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->labelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->referenceOfLabelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->modificationFlagOnDevice), sizeof(bool)),isInitSpaceReady);

	allObjects = static_cast<FourLimits*>(malloc(sizeof(FourLimits) * width * height));
	allObjectRects = static_cast<ObjectRect*>(malloc(sizeof(ObjectRect) * width * height));
	return isInitSpaceReady;
}

inline void Detector::CopyFrameData(unsigned char* frame)
{
	this->isFrameDataReady = true;

	memcpy(this->originalFrameOnHost, frame, sizeof(unsigned char) * width * height);
	memset(this->allObjects, -1, sizeof(FourLimits) * width * height);
	memset(this->allObjectRects, 0, sizeof(ObjectRect) * width * height);

	CheckCUDAReturnStatus(cudaMemcpy(this->originalFrameOnDevice, this->originalFrameOnHost, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice), isFrameDataReady);
	if(isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Copy current frame data failed!", LogLevel::Error);
	}
}

inline void Detector::GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height)
{
	// top
	for (auto r = 0; r < height; ++r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].top == -1)
				allObjects[label].top = r;
		}
	}
	// bottom
	for (auto r = height - 1; r >= 0; --r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].bottom == -1)
				allObjects[label].bottom = r;
			if (allObjects[label].bottom - allObjects[label].top + 1 < 2)
				allObjects[label].top = -1;
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
	for (auto c = width - 1; c >= 0; --c)
	{
		for (auto r = 0; r < height; ++r)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].right == -1)
				allObjects[label].right = c;
			if (allObjects[label].right - allObjects[label].left + 1 < 2)
				allObjects[label].top = -1;
		}
	}
}

inline void Detector::ConvertFourLimitsToRect(FourLimits* allObjects, ObjectRect* allObjectRects, int width, int height)
{
	for (auto i = 0; i < width * height; ++i)
	{
		if (allObjects[i].top == -1)
			continue;
		allObjectRects[i].width = allObjects[i].right - allObjects[i].left + 1;
		allObjectRects[i].height = allObjects[i].bottom - allObjects[i].top + 1;
		allObjectRects[i].lt = Point(allObjects[i].left, allObjects[i].top);
		allObjectRects[i].rb = Point(allObjects[i].right, allObjects[i].bottom);
	}
}

inline bool Detector::CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const
{
	auto firstCenterX = (objectFirst.right + objectFirst.left) / 2;
	auto firstCenterY = (objectFirst.bottom + objectFirst.top) / 2;

	auto secondCenterX = (objectSecond.right + objectSecond.left) / 2;
	auto secondCenterY = (objectSecond.bottom + objectSecond.top) / 2;

	auto firstWidth = objectFirst.right - objectFirst.left + 1;
	auto firstHeight = objectFirst.bottom - objectFirst.top + 1;

	auto secondWidth = objectSecond.right - objectSecond.left + 1;
	auto secondHeight = objectSecond.bottom - objectSecond.top + 1;

	auto centerXDiff = std::abs(firstCenterX - secondCenterX);
	auto centerYDiff = std::abs(firstCenterY - secondCenterY);

	if (centerXDiff <= (firstWidth + secondWidth) / 2 + 1 && centerYDiff <= (firstHeight + secondHeight) / 2 + 1)
		return true;

	return false;
}

inline void Detector::MergeObjects() const
{
	for (auto i = 0; i < width * height; ++i)
	{
		if (allObjects[i].top == -1)
			continue;
		for (auto j = 0; j < width * height; ++j)
		{
			if (i == j || allObjects[j].top == -1)
				continue;
			if (CheckCross(allObjects[i], allObjects[j]))
			{
				allObjects[j].top = -1;

				if (allObjects[i].top > allObjects[j].top)
					allObjects[i].top = allObjects[j].top;

				if (allObjects[i].left > allObjects[j].left)
					allObjects[i].left = allObjects[j].left;

				if (allObjects[i].right < allObjects[j].right)
					allObjects[i].right = allObjects[j].right;

				if (allObjects[i].bottom < allObjects[j].bottom)
					allObjects[i].bottom = allObjects[j].bottom;
			}

			if ((allObjects[i].bottom - allObjects[i].top) > TARGET_HEIGHT_MAX_LIMIT ||
				(allObjects[i].right - allObjects[i].left) > TARGET_WIDTH_MAX_LIMIT)
			{
				allObjects[i].top = -1;
				break;
			}
		}
	}
}

inline void Detector::RemoveObjectWithLowContrast() const
{
	for (auto i = 0; i < width * height; ++i)
	{
		if(allObjects[i].top == -1)
			continue;

		unsigned char averageValue = 0;
		unsigned char centerValue = 0;

		auto objectWidth = allObjects[i].right - allObjects[i].left + 1;
		auto objectHeight = allObjects[i].bottom - allObjects[i].top + 1;

		auto surroundBoxWidth = 3 * objectWidth;
		auto surroundBoxHeight = 3 * objectHeight;

		auto centerX = (allObjects[i].right + allObjects[i].left) / 2;
		auto centerY = (allObjects[i].bottom + allObjects[i].top) / 2;

		auto leftTopX = centerX - surroundBoxWidth / 2;
		if (leftTopX < 0)
		{
			leftTopX = 0;
		}

		auto leftTopY = centerY - surroundBoxHeight / 2;
		if (leftTopY < 0)
		{
			leftTopY = 0;
		}

		auto rightBottomX = leftTopX + surroundBoxWidth;
		if (rightBottomX >= width)
		{
			rightBottomX = width - 1;
		}

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY >= height)
		{
			rightBottomY = height - 1;
		}

		FourLimits surroundingBox(leftTopY,rightBottomY, leftTopX, rightBottomX);\

		Util::CalculateAverage(discretizationResultOnHost, surroundingBox, averageValue, objectWidth);

		Util::CalCulateCenterValue(discretizationResultOnHost, centerValue, objectWidth, centerX, centerY);

		if (std::abs(static_cast<int>(centerValue) - static_cast<int>(averageValue)) < 3)
		{
			allObjects[i].top = -1;
		}
	}
}

inline void Detector::DetectTargets(unsigned char* frame)
{
	CopyFrameData(frame);

	if(isFrameDataReady == true)
	{
		// dilation on gpu
		DilationFilter(this->originalFrameOnDevice, this->dilationResultOnDevice, width, height, radius);

		// level disretization on gpu
		LevelDiscretizationOnGPU(this->dilationResultOnDevice, width, height, discretizationScale);

		// CCL On Device
		MeshCCL(this->dilationResultOnDevice, this->labelsOnDevice, this->referenceOfLabelsOnDevice, this->modificationFlagOnDevice, width, height);

		// copy labels from device to host
		cudaMemcpy(this->labelsOnHost, this->labelsOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->discretizationResultOnHost, this->dilationResultOnDevice, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

		// get all object
		GetAllObjects(labelsOnHost, allObjects, width, height);

		// convert all obejct to rect
		ConvertFourLimitsToRect(allObjects, allObjectRects, width, height);

		// show result
//		ShowFrame::DrawRectangles(originalFrameOnHost, allObjectRects, width, height);

		// Merge all objects
//		MergeObjects();

		// Remove objects with low contrast
		RemoveObjectWithLowContrast();

	}
}
#endif
