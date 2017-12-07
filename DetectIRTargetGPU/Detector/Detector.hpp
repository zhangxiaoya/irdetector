#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <cuda_runtime_api.h>
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../Headers/GlobalMainHeaders.h"
#include "../Headers/DetectorParameters.h"
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
#include "../Models/DetectedTarget.hpp"

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

	void GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height);

	void ConvertFourLimitsToRect(FourLimits* allObjects, ObjectRect* allObjectRects, int width, int height, int validObjectCount = 0);

	bool CheckCross(const FourLimits& objectFirst, const FourLimits& objectSecond) const;

	bool CheckCross(const DetectedTarget& objectFirst, const DetectedTarget& objectSecond) const;

	void MergeObjects();

	void RemoveObjectWithLowContrast();

	void RemoveInValidObjects();

	void RemoveInvalidObjectAfterMerge();

	void FalseAlarmFilter();

protected:
	bool ReleaseSpace();

	void InitForbiddenZones();

	bool IsInForbiddenZone(const FourLimits& candidateTargetRegion) const;

	bool IsAtBorderZone(const FourLimits& candidateTargetRegion) const;

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
	ObjectRect* allObjectRects;
	FourLimitsWithScore* insideObjects;
	DetectedTarget* allObjectWithProp;

	FourLimits ForbiddenZones[MAX_FORBIDDEN_ZONE_COUNT];
	int ForbiddenZoneCount;

	int validObjectsCount;
	int lastResultCount;

	int TargetWidthMaxLimit;
	int TargetHeightMaxLimit;

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
	  allObjectRects(nullptr),
	  insideObjects(nullptr),
	  allObjectWithProp(nullptr),
	  ForbiddenZoneCount(0),
	  validObjectsCount(0),
	  lastResultCount(0),
	  TargetWidthMaxLimit(TARGET_WIDTH_MAX_LIMIT),
	  TargetHeightMaxLimit(TARGET_HEIGHT_MAX_LIMIT),
	  CHECK_ORIGIN_FLAG(false),
	  CHECK_DECRETIZATED_FLAG(false),
	  CHECK_SURROUNDING_BOUNDARY_FLAG(false),
	  CHECK_INSIDE_BOUNDARY_FLAG(false),
	  CHECK_COVERAGE_FLAG(false),
	  CHECK_STANDARD_DEVIATION_FLAG(false)
{
	InitForbiddenZones();
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
	if(this->insideObjects != nullptr)
	{
		delete[] insideObjects;
	}
	if(this->allObjectWithProp != nullptr)
	{
		delete[] allObjectWithProp;
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

// Manul set Forbidden Zone, sine the bad-point of camera
inline void Detector::InitForbiddenZones()
{
	ForbiddenZoneCount = 1;

	ForbiddenZones[0].top = 101;
	ForbiddenZones[0].bottom = 106;
	ForbiddenZones[0].left = 289;
	ForbiddenZones[0].right = 295;
}

inline bool Detector::IsInForbiddenZone(const FourLimits& candidateTargetRegion) const
{
	double centerX = (candidateTargetRegion.left + candidateTargetRegion.right) / 2.0;
	double centerY = (candidateTargetRegion.top + candidateTargetRegion.bottom) / 2.0;
	for (auto i = 0; i < ForbiddenZoneCount; ++i)
	{
		if (centerX > ForbiddenZones[i].left && centerX < ForbiddenZones[i].right && centerY > ForbiddenZones[i].top && centerY < ForbiddenZones[i].bottom)
			return true;
	}
	return false;
}

inline bool Detector::IsAtBorderZone(const FourLimits& candidateTargetRegion) const
{
	if (candidateTargetRegion.left < 3
		|| candidateTargetRegion.bottom > (Height - 4)
		|| candidateTargetRegion.top < 3
		|| candidateTargetRegion.right > (Width - 4))
		return true;
	return false;
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
	insideObjects = static_cast<FourLimitsWithScore*>(malloc(sizeof(FourLimitsWithScore) * Width * Height));
	allObjectWithProp = static_cast<DetectedTarget*>(malloc(sizeof(DetectedTarget) *Width * Height));
	return isInitSpaceReady;
}

inline void Detector::CopyFrameData(unsigned short* frame)
{
	this->isFrameDataReady = true;

	memcpy(this->originalFrameOnHost, frame, sizeof(unsigned short) * Width * Height);
	memset(this->originalFrameOnHost, MAX_PIXEL_VALUE, FRAME_HEADER_LENGTH);
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
	for (auto r = 0; r < height; ++r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].top == -1)
			{
				allObjects[label].top = r;
				allObjectWithProp[label].fourLimits.top = r;
			}
			if (allObjects[label].bottom < r)
			{
				allObjects[label].bottom = r;
				allObjectWithProp[label].fourLimits.bottom = r;
				allObjectWithProp[label].height = allObjectWithProp[label].fourLimits.bottom - allObjectWithProp[label].fourLimits.top + 1;
				allObjectWithProp[label].centerY = (allObjectWithProp[label].fourLimits.bottom + allObjectWithProp[label].fourLimits.top) / 2;
			}
			if(allObjects[label].left == -1)
			{
				allObjects[label].left = c;
				allObjectWithProp[label].fourLimits.left = c;
			}
			else if (allObjects[label].left > c)
			{
				allObjects[label].left = c;
				allObjectWithProp[label].fourLimits.left = c;
			}
			if (allObjects[label].right < c)
			{
				allObjects[label].right = c;
				allObjectWithProp[label].fourLimits.right = c;
				allObjectWithProp[label].width = allObjectWithProp[label].fourLimits.right - allObjectWithProp[label].fourLimits.left + 1;
				allObjectWithProp[label].centerX = (allObjectWithProp[label].fourLimits.left + allObjectWithProp[label].fourLimits.right) / 2;
			}
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

inline bool Detector::CheckCross(const DetectedTarget& objectFirst, const DetectedTarget& objectSecond) const
{
	auto centerXDiff = std::abs(objectFirst.centerX - objectSecond.centerX);
	auto centerYDiff = std::abs(objectFirst.centerY - objectSecond.centerY);

	if (centerXDiff <= ((objectFirst.width + objectSecond.width) / 2 + 1) && centerYDiff <= ((objectFirst.height + objectSecond.height) / 2 + 1))
	{
		return true;
	}
	return false;
}

inline void Detector::MergeObjects()
{
#pragma omp parallel
	for (auto i = 0; i < validObjectsCount; ++i)
	{
		if (allObjects[i].top == -1)
			continue;
		for (auto j = 0; j < validObjectsCount; ++j)
		{
			if (i == j || allObjects[j].top == -1)
				continue;
			//if (CheckCross(allObjects[i], allObjects[j]))
			if (CheckCross(allObjectWithProp[i], allObjectWithProp[j]))
			{
				if (allObjects[i].top > allObjects[j].top)
					allObjects[i].top = allObjects[j].top;

				if (allObjects[i].left > allObjects[j].left)
					allObjects[i].left = allObjects[j].left;

				if (allObjects[i].right < allObjects[j].right)
					allObjects[i].right = allObjects[j].right;

				if (allObjects[i].bottom < allObjects[j].bottom)
					allObjects[i].bottom = allObjects[j].bottom;

				if (allObjectWithProp[i].fourLimits.top > allObjectWithProp[j].fourLimits.top)
					allObjectWithProp[i].fourLimits.top = allObjectWithProp[j].fourLimits.top;

				if (allObjectWithProp[i].fourLimits.left > allObjectWithProp[j].fourLimits.left)
					allObjectWithProp[i].fourLimits.left = allObjectWithProp[j].fourLimits.left;

				if (allObjectWithProp[i].fourLimits.right < allObjectWithProp[j].fourLimits.right)
					allObjectWithProp[i].fourLimits.right = allObjectWithProp[j].fourLimits.right;

				if (allObjectWithProp[i].fourLimits.bottom < allObjectWithProp[j].fourLimits.bottom)
					allObjectWithProp[i].fourLimits.bottom = allObjectWithProp[j].fourLimits.bottom;

				allObjects[j].top = -1;
				allObjectWithProp[j].width = -1;

			}

			//if ((allObjects[i].bottom - allObjects[i].top + 1) > TargetHeightMaxLimit ||
			//	(allObjects[i].right - allObjects[i].left + 1) > TargetWidthMaxLimit)
			if(allObjectWithProp[i].height > TargetHeightMaxLimit ||
				allObjectWithProp[i].width > TargetWidthMaxLimit)
			{
				allObjects[i].top = -1;
				allObjectWithProp[i].width = -1;
				break;
			}
		}
		// ConvertFourLimitsToRect(allObjects, allObjectRects, Width, Height, validObjectsCount);
		// ShowFrame::DrawRectangles(originalFrameOnHost, allObjectRects, Width, Height);
	}
}

inline void Detector::RemoveObjectWithLowContrast()
{
	for (auto i = 0; i < validObjectsCount; ++i)
	{
		if (allObjects[i].top == -1)
			continue;

		unsigned short averageValue = 0;
		unsigned short centerValue = 0;

		// auto objectWidth = allObjects[i].right - allObjects[i].left + 1;
		// auto objectHeight = allObjects[i].bottom - allObjects[i].top + 1;
		auto objectWidth = allObjectWithProp[i].width;
		auto objectHeight = allObjectWithProp[i].height;

		auto surroundBoxWidth = 3 * objectWidth;
		auto surroundBoxHeight = 3 * objectHeight;

		// auto centerX = (allObjects[i].right + allObjects[i].left) / 2;
		// auto centerY = (allObjects[i].bottom + allObjects[i].top) / 2;
		auto centerX = allObjectWithProp[i].centerX;
		auto centerY = allObjectWithProp[i].centerY;

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
			allObjects[i].top = -1;
			allObjectWithProp[i].width = -1;
		}
		// ConvertFourLimitsToRect(allValidObjects, allObjectRects, Width, Height, validObjectsCount);
		// ShowFrame::DrawRectangles(originalFrameOnHost, allObjectRects, Width, Height);
	}
}

inline void Detector::RemoveInValidObjects()
{
	int oldValidObjectCount = validObjectsCount;
	validObjectsCount = 0;
	for (auto i = 0; i < oldValidObjectCount; ++i)
	{
		if(allObjects[i].top == -1)
			continue;
		// if(allObjects[i].bottom - allObjects[i].top + 1 > TargetHeightMaxLimit || allObjects[i].right - allObjects[i].left + 1 > TargetWidthMaxLimit)
		if (allObjectWithProp[i].height > TargetHeightMaxLimit || allObjectWithProp[i].width > TargetWidthMaxLimit)
			continue;
		// if(allObjects[i].bottom - allObjects[i].top + 1 < 1 || allObjects[i].right - allObjects[i].left + 1 < 1)
		if (allObjectWithProp[i].height < 1 || allObjectWithProp[i].width < 1)
			continue;

		allObjects[validObjectsCount] = allObjects[i];
		allObjectWithProp[validObjectsCount] = allObjectWithProp[i];
		validObjectsCount++;
	}
}

inline void Detector::RemoveInvalidObjectAfterMerge()
{
	auto newValidaObjectCount = 0;
	for (auto i = 0; i < validObjectsCount;)
	{
		if (allObjects[i].top == -1)
		{
			i++;
			continue;
		}
		if(IsInForbiddenZone(allObjects[i]) == true)
		{
			i++;
			continue;
		}
		if(IsAtBorderZone(allObjects[i]) == true)
		{
			i++;
			continue;
		}
		allObjects[newValidaObjectCount] = allObjects[i];
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
		auto object = allObjects[i];
		filters.InitObjectParameters(originalFrameOnHost, discretizationResultOnHost, object, Width, Height);

		auto currentResult = (CHECK_ORIGIN_FLAG && filters.CheckOriginalImageSuroundedBox(originalFrameOnHost, Width, Height))
			|| (CHECK_DECRETIZATED_FLAG && filters.CheckDiscretizedImageSuroundedBox(discretizationResultOnHost, Width, Height));
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
			if(contrast < FALSE_ALARM_FILTER_MIN_CONTRAST)
				continue;
//			this->insideObjects[lastResultCount].score = score + static_cast<int>(filters.GetCenterValue());
			this->insideObjects[lastResultCount].score = score + contrast;
			lastResultCount++;
		}
	}

	if (lastResultCount >= MAX_DETECTED_TARGET_COUNT)
		std::sort(this->insideObjects, this->insideObjects + lastResultCount, Util::CompareResult);
}

inline void Detector::DetectTargets(unsigned short* frame, DetectResultSegment* result, FourLimits** allCandidatesTargets, int* allCandidateTargetsCount)
{
	CopyFrameData(frame);

	if (isFrameDataReady == true)
	{
		// assume all object(pixel) all valid
		validObjectsCount = Width * Height;

		// dilation on gpu
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

		// Remove objects with low contrast
		RemoveObjectWithLowContrast();

		// remove invalid objects
		RemoveInValidObjects();

		// Merge all objects
		MergeObjects();
		// Remove objects after merge
		RemoveInvalidObjectAfterMerge();

		// Copy frame header
		memcpy(result->header, frame, FRAME_HEADER_LENGTH);

		// return all candidate targets before remove false alarm
		if (allCandidatesTargets != nullptr)
			*allCandidatesTargets = this->allObjects;
		if (allCandidateTargetsCount != nullptr)
			*allCandidateTargetsCount = this->validObjectsCount;

		// Filter all candiates
		FalseAlarmFilter();

		// put all valid result to resultSegment
		result->targetCount = lastResultCount >= MAX_DETECTED_TARGET_COUNT ? MAX_DETECTED_TARGET_COUNT : lastResultCount;

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
