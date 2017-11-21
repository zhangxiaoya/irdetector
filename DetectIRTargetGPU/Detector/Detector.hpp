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
#include "../Monitor/Filter.hpp"
#include "../Models/DetectResultSegment.hpp"
#include "../Models/FourLimitsWithScore.hpp"

inline bool CompareResult(FourLimitsWithScore& a, FourLimitsWithScore& b)
{
	return a.score - b.score > 0.0000001;
}

class Detector
{
public:
	Detector(const int width, const int height, const int dilationRadius, const int discretizationScale);

	~Detector();

	bool InitSpace();

	void DetectTargets(unsigned short* frame, DetectResultSegment* result, FourLimits** allCandidatesTargets = nullptr, int* allCandidateTargetsCount = nullptr);

	void SetRemoveFalseAlarmParameters(bool checkStandardDeviationFlag,
	                                   bool checkSurroundingBoundaryFlag,
	                                   bool checkInsideBoundaryFlag,
	                                   bool checkCoverageFlag,
	                                   bool checkOriginalImageThreshold,
	                                   bool checkDiscretizatedThreshold);

private:
	void CopyFrameData(unsigned short* frame);

	static void GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height);

	static void ConvertFourLimitsToRect(FourLimits* allObjects, ObjectRect* allObjectRects, int width, int height, int validObjectCount = 0);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	void MergeObjects() const;

	void RemoveObjectWithLowContrast() const;

	void RemoveInValidObjects();

	void RemoveInvalidObjectAfterMerge();

	void FalseAlarmFilter();

protected:
	bool ReleaseSpace();

private:
	int Width;
	int Height;

	int DilationRadius;
	int DiscretizationScale;

	bool isInitSpaceReady;
	bool isFrameDataReady;

	unsigned short* originalFrameOnHost;
	unsigned short* originalFrameOnDevice;
	unsigned short* dilationResultOnDevice;
	unsigned short* discretizationResultOnHost;
	unsigned short* tempFrame;

	int* labelsOnHost;
	int* labelsOnDevice;
	int* referenceOfLabelsOnDevice;

	bool* modificationFlagOnDevice;

	FourLimits* allObjects;
	FourLimits* allValidObjects;
	ObjectRect* allObjectRects;
	FourLimitsWithScore* insideObjects;

	int validObjectsCount;
	int lastResultCount;

	int TARGET_WIDTH_MAX_LIMIT;
	int TARGET_HEIGHT_MAX_LIMIT;

	Filter filters;

	bool CHECK_ORIGIN_FLAG;
	bool CHECK_DECRETIZATED_FLAG;
	bool CHECK_SURROUNDING_BOUNDARY_FLAG;
	bool CHECK_INSIDE_BOUNDARY_FLAG;
	bool CHECK_COVERAGE_FLAG;
	bool CHECK_STANDARD_DEVIATION_FLAG;
};

inline Detector::Detector(const int width, const int height, const int dilationRadius, const int discretizationScale)
	: Width(width),
	  Height(height),
	  DilationRadius(dilationRadius),
	  DiscretizationScale(discretizationScale),
	  isInitSpaceReady(true),
	  isFrameDataReady(true),
	  originalFrameOnHost(nullptr),
	  originalFrameOnDevice(nullptr),
	  dilationResultOnDevice(nullptr),
	  discretizationResultOnHost(nullptr),
	  tempFrame(nullptr),
	  labelsOnHost(nullptr),
	  labelsOnDevice(nullptr),
	  referenceOfLabelsOnDevice(nullptr),
	  modificationFlagOnDevice(nullptr),
	  allObjects(nullptr),
	  allValidObjects(nullptr),
	  allObjectRects(nullptr),
	  insideObjects(nullptr),
	  validObjectsCount(0),
	  lastResultCount(0),
	  TARGET_WIDTH_MAX_LIMIT(20),
	  TARGET_HEIGHT_MAX_LIMIT(20),
	  CHECK_ORIGIN_FLAG(false),
	  CHECK_DECRETIZATED_FLAG(false),
	  CHECK_SURROUNDING_BOUNDARY_FLAG(false),
	  CHECK_INSIDE_BOUNDARY_FLAG(false),
	  CHECK_COVERAGE_FLAG(false),
	  CHECK_STANDARD_DEVIATION_FLAG(false)
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
	if (this->labelsOnHost != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->labelsOnHost), status);
		if (status == true)
		{
			this->labelsOnHost = nullptr;
		}
	}
	if (this->discretizationResultOnHost != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->discretizationResultOnHost), status);
		if (status == true)
		{
			this->discretizationResultOnHost = nullptr;
		}
	}
	if (this->tempFrame != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->tempFrame), status);
		if (status == true)
		{
			this->tempFrame = nullptr;
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
			this->dilationResultOnDevice = nullptr;
		}
	}
	if (this->labelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->labelsOnDevice), status);
		if (status == true)
		{
			this->labelsOnDevice = nullptr;
		}
	}
	if (this->referenceOfLabelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->referenceOfLabelsOnDevice), status);
		if (status == true)
		{
			this->referenceOfLabelsOnDevice = nullptr;
		}
	}
	if (this->modificationFlagOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->modificationFlagOnDevice), status);
		if (status == true)
		{
			this->modificationFlagOnDevice = nullptr;
		}
	}

	if (this->allObjects != nullptr)
	{
		delete[] allObjects;
	}
	if (this->allObjectRects != nullptr)
	{
		delete[] allObjectRects;
	}
	if (this->allValidObjects != nullptr)
	{
		delete[] allValidObjects;
	}
	if(this->insideObjects != nullptr)
	{
		delete[] insideObjects;
	}

	if (status == true)
	{
		logPrinter.PrintLogs("Release space success!", Info);
	}
	else
	{
		logPrinter.PrintLogs("Release space failed!", Error);
	}
	return status;
}

inline bool Detector::InitSpace()
{
	logPrinter.PrintLogs("Release space before re-init space ...", Info);
	if (ReleaseSpace() == false)
		return false;

	isInitSpaceReady = true;
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->originalFrameOnHost), sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->labelsOnHost), sizeof(int) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->discretizationResultOnHost), sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->tempFrame), sizeof(unsigned short) * Width * Height), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->originalFrameOnDevice), sizeof(unsigned short) * Width * Height),isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->dilationResultOnDevice), sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->labelsOnDevice), sizeof(int) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->referenceOfLabelsOnDevice), sizeof(int) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->modificationFlagOnDevice), sizeof(bool)),isInitSpaceReady);

	allObjects = static_cast<FourLimits*>(malloc(sizeof(FourLimits) * Width * Height));
	allObjectRects = static_cast<ObjectRect*>(malloc(sizeof(ObjectRect) * Width * Height));
	allValidObjects = static_cast<FourLimits*>(malloc(sizeof(FourLimits) * Width * Height));
	insideObjects = static_cast<FourLimitsWithScore*>(malloc(sizeof(FourLimitsWithScore) * Width * Height));
	return isInitSpaceReady;
}

inline void Detector::CopyFrameData(unsigned short* frame)
{
	this->isFrameDataReady = true;

	memcpy(this->originalFrameOnHost, frame, sizeof(unsigned short) * Width * Height);
	memset(this->originalFrameOnHost, 65535, 16);
	memset(this->allObjects, -1, sizeof(FourLimits) * Width * Height);
	memset(this->allObjectRects, 0, sizeof(ObjectRect) * Width * Height);

	CheckCUDAReturnStatus(cudaMemcpy(this->originalFrameOnDevice, this->originalFrameOnHost, sizeof(unsigned short) * Width * Height, cudaMemcpyHostToDevice), isFrameDataReady);
	if (isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Copy current frame data failed!", Error);
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

inline void Detector::ConvertFourLimitsToRect(FourLimits* allObjects, ObjectRect* allObjectRects, int width, int height, int validObjectCount)
{
	if (validObjectCount == 0)
		validObjectCount = width * height;
	for (auto i = 0; i < validObjectCount; ++i)
	{
		if (allObjects[i].top == -1)
		{
			allObjectRects[i].width = 0;
			continue;
		}
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
	for (auto i = 0; i < validObjectsCount; ++i)
	{
		if (allValidObjects[i].top == -1)
			continue;
		for (auto j = 0; j < validObjectsCount; ++j)
		{
			if (i == j || allValidObjects[j].top == -1)
				continue;
			if (CheckCross(allValidObjects[i], allValidObjects[j]))
			{
				if (allValidObjects[i].top > allValidObjects[j].top)
					allValidObjects[i].top = allValidObjects[j].top;

				if (allValidObjects[i].left > allValidObjects[j].left)
					allValidObjects[i].left = allValidObjects[j].left;

				if (allValidObjects[i].right < allValidObjects[j].right)
					allValidObjects[i].right = allValidObjects[j].right;

				if (allValidObjects[i].bottom < allValidObjects[j].bottom)
					allValidObjects[i].bottom = allValidObjects[j].bottom;

				allValidObjects[j].top = -1;

			}

			if ((allValidObjects[i].bottom - allValidObjects[i].top + 1) > TARGET_HEIGHT_MAX_LIMIT ||
				(allValidObjects[i].right - allValidObjects[i].left + 1) > TARGET_WIDTH_MAX_LIMIT)
			{
				allValidObjects[i].top = -1;
				break;
			}
		}
//		ConvertFourLimitsToRect(allValidObjects, allObjectRects, width, height, validObjectsCount);
//		ShowFrame::DrawRectangles(originalFrameOnHost, allObjectRects, width, height);
	}
}

inline void Detector::RemoveObjectWithLowContrast() const
{
	for (auto i = 0; i < validObjectsCount; ++i)
	{
		if (allValidObjects[i].top == -1)
			continue;

		unsigned short averageValue = 0;
		unsigned short centerValue = 0;

		auto objectWidth = allValidObjects[i].right - allValidObjects[i].left + 1;
		auto objectHeight = allValidObjects[i].bottom - allValidObjects[i].top + 1;

		if (objectHeight < 2 || objectWidth < 2 || objectHeight > 20 || objectWidth > 20)
		{
			allValidObjects[i].top = -1;
			continue;
		}

		auto surroundBoxWidth = 3 * objectWidth;
		auto surroundBoxHeight = 3 * objectHeight;

		auto centerX = (allValidObjects[i].right + allValidObjects[i].left) / 2;
		auto centerY = (allValidObjects[i].bottom + allValidObjects[i].top) / 2;

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
		if (rightBottomX >= Width)
		{
			rightBottomX = Width - 1;
		}

		auto rightBottomY = leftTopY + surroundBoxHeight;
		if (rightBottomY >= Height)
		{
			rightBottomY = Height - 1;
		}

		FourLimits surroundingBox(leftTopY, rightBottomY, leftTopX, rightBottomX);

		Util::CalculateAverage(discretizationResultOnHost, surroundingBox, averageValue, objectWidth);

		Util::CalCulateCenterValue(discretizationResultOnHost, centerValue, objectWidth, centerX, centerY);

		if (std::abs(static_cast<int>(centerValue) - static_cast<int>(averageValue)) < 3)
		{
			allValidObjects[i].top = -1;
		}
	}
}

inline void Detector::RemoveInValidObjects()
{
	validObjectsCount = 0;
	for (auto i = 0; i < Width * Height; ++i)
	{
		if (allObjects[i].top != -1 && ((allObjects[i].right - allObjects[i].left + 1) > 3 || (allObjects[i].bottom - allObjects[i].top + 1 > 3)))
		{
			allValidObjects[validObjectsCount] = allObjects[i];
			validObjectsCount++;
		}
	}
}

inline void Detector::RemoveInvalidObjectAfterMerge()
{
	auto newValidaObjectCount = 0;
	for (auto i = 0; i < validObjectsCount;)
	{
		if (allValidObjects[i].top == -1)
		{
			i++;
			continue;
		}
		allValidObjects[newValidaObjectCount] = allValidObjects[i];
		++i;
		newValidaObjectCount++;
	}
	validObjectsCount = newValidaObjectCount;
}

inline void Detector::FalseAlarmFilter()
{
	lastResultCount = 0;

	for (auto i = 0; i < validObjectsCount; ++i)
	{
		auto score = 0;
		auto object = allValidObjects[i];
		filters.InitObjectParameters(originalFrameOnHost, discretizationResultOnHost, object, Width, Height);

		auto currentResult = (CHECK_ORIGIN_FLAG && filters.CheckOriginalImageSuroundedBox(originalFrameOnHost, Width, Height, object))
			|| (CHECK_DECRETIZATED_FLAG && filters.CheckDiscretizedImageSuroundedBox(discretizationResultOnHost, Width, Height, object));
		if (currentResult == false) continue;
		score++;

		if (CHECK_SURROUNDING_BOUNDARY_FLAG)
		{
			currentResult &= filters.CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(discretizationResultOnHost, Width, Height, object);
			if (currentResult == false) continue;
			score++;
		}
		if (CHECK_COVERAGE_FLAG)
		{
			currentResult &= filters.CheckCoverageOfPreprocessedFrame(discretizationResultOnHost, Width, object);
			if (currentResult == false) continue;
			score++;
		}
		if (CHECK_INSIDE_BOUNDARY_FLAG)
		{
			currentResult &= filters.CheckInsideBoundaryDescendGradient(originalFrameOnHost, Width, object);
			if (currentResult == false) continue;
			score++;
		}
		if (CHECK_STANDARD_DEVIATION_FLAG)
		{
			currentResult &= filters.CheckStandardDeviation(originalFrameOnHost, Width, object);
			if (currentResult == false) continue;
			score++;
		}
		if (currentResult != true)
			object.top = -1;
		else
		{
			this->insideObjects[lastResultCount].object = object;
			auto contrast = filters.GetContrast();
			if(contrast < 1.002)
				continue;
//			this->insideObjects[lastResultCount].score = score + static_cast<int>(filters.GetCenterValue());
			this->insideObjects[lastResultCount].score = score + contrast;
			lastResultCount++;
		}
	}

	if (lastResultCount >= 5)
		std::sort(this->insideObjects, this->insideObjects + lastResultCount, CompareResult);
}

inline void Detector::DetectTargets(unsigned short* frame, DetectResultSegment* result, FourLimits** allCandidatesTargets, int* allCandidateTargetsCount)
{
	CopyFrameData(frame);

	if (isFrameDataReady == true)
	{
		// dilation on gpu
//		DilationFilter(this->originalFrameOnDevice, this->dilationResultOnDevice, width, height, DilationRadius);
		NaiveDilation(this->originalFrameOnDevice, this->dilationResultOnDevice, Width, Height, DilationRadius);

		// level disretization on gpu
		LevelDiscretizationOnGPU(this->dilationResultOnDevice, Width, Height, DiscretizationScale);

		// CCL On Device
		MeshCCL(this->dilationResultOnDevice, this->labelsOnDevice, this->referenceOfLabelsOnDevice, this->modificationFlagOnDevice, Width, Height);

		// copy labels from device to host
		cudaMemcpy(this->labelsOnHost, this->labelsOnDevice, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost);
		cudaMemcpy(this->discretizationResultOnHost, this->dilationResultOnDevice, sizeof(unsigned short) * Width * Height, cudaMemcpyDeviceToHost);

		// get all object
		GetAllObjects(labelsOnHost, allObjects, Width, Height);

		// remove invalid objects
		RemoveInValidObjects();

//		auto frameIndex = reinterpret_cast<unsigned*>(frame);
//
//		auto temp = static_cast<double>((static_cast<double>(*frameIndex) / 1250)) + static_cast<double>(25);
//		if (temp >= 360.0)
//			temp -= 360.0;
//		if (temp < 0)
//			temp += 360.0;

//		std::cout <<"---------------------------------------------------->"<< temp << std::endl;
//		if(temp > 4.0 && temp < 6.0)
//		{
//			// convert all obejct to rect
//			ConvertFourLimitsToRect(allObjects, allObjectRects, width, height);
//
//			// show result
//			ShowFrame::DrawRectangles(originalFrameOnHost, allObjectRects, width, height);
//		}


		// Merge all objects
		MergeObjects();

		MergeObjects();

		// Remove objects with low contrast
//		RemoveObjectWithLowContrast();

		// Remove objects after merge
		RemoveInvalidObjectAfterMerge();

		// Copy frame header
		memcpy(result->header, frame, 16);

		// return all candidate targets before remove false alarm
		if (allCandidatesTargets != nullptr)
			*allCandidatesTargets = this->allValidObjects;
		if (allCandidateTargetsCount != nullptr)
			*allCandidateTargetsCount = this->validObjectsCount;

		// Filter all candiates
		FalseAlarmFilter();

		// put all valid result to resultSegment
		result->targetCount = lastResultCount >= 3 ? 3 : lastResultCount;

		for (auto i = 0; i < result->targetCount; ++i)
		{
			TargetPosition pos;
			pos.topLeftX = insideObjects[i].object.left;
			pos.topLeftY = insideObjects[i].object.top;
			pos.bottomRightX = insideObjects[i].object.right;
			pos.bottomRightY = insideObjects[i].object.bottom;
			result->targets[i] = pos;
		}
	}
}

inline void Detector::SetRemoveFalseAlarmParameters(const bool checkStandardDeviationFlag,
                                                    const bool checkSurroundingBoundaryFlag,
                                                    const bool checkInsideBoundaryFlag,
                                                    const bool checkCoverageFlag,
                                                    const bool checkOriginalImageThreshold,
                                                    const bool checkDiscretizatedThreshold)
{
	CHECK_STANDARD_DEVIATION_FLAG = checkStandardDeviationFlag;
	CHECK_SURROUNDING_BOUNDARY_FLAG = checkSurroundingBoundaryFlag;
	CHECK_INSIDE_BOUNDARY_FLAG = checkInsideBoundaryFlag;
	CHECK_COVERAGE_FLAG = checkCoverageFlag;

	CHECK_ORIGIN_FLAG = checkOriginalImageThreshold;
	filters.SetConvexPartitionOfOriginalImage(20 * 256);
	filters.SetConcavePartitionOfOriginalImage(1);

	CHECK_DECRETIZATED_FLAG = checkDiscretizatedThreshold;
	filters.SetConvexPartitionOfDiscretizedImage(20 * 256);
	filters.SetConcavePartitionOfDiscretizedImage(1);
}
#endif
