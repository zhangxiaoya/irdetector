#ifndef __FILTER_H__
#define __FILTER_H__
#include "../Models/FourLimits.h"
#include "../Common/Util.h"
#include <cmath>
#include "../Headers/GlobalMainHeaders.h"

class Filter
{
public:
	static bool CheckOriginalImageSuroundedBox(unsigned char* originalFrameOnHost, int width, int height, const FourLimits& object);

	static bool CheckDiscretizedImageSuroundedBox(unsigned char* discretizatedFrameOnHost, int width, int height, const FourLimits& object);

	static bool CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* frameOnHost, int width, const FourLimits& object);

	static bool CheckCoverageOfPreprocessedFrame(unsigned char* frameOnHost, int width, const FourLimits& object);

	static bool CheckInsideBoundaryDescendGradient(unsigned char* frameOnHost, int width, const FourLimits& object);

	static bool CheckStandardDeviation(unsigned char* frameOnHost, int width, const FourLimits& object);

private:
	static bool CheckPeakValueAndAverageValue(unsigned char* frameOnHost, int width, int height, const FourLimits& object, int convexPartition, int concavePartition);
};

inline bool Filter::CheckPeakValueAndAverageValue(unsigned char* frameOnHost, int width, int height, const FourLimits& object, int convexPartition, int concavePartition)
{
	auto centerX = (object.left + object.right) / 2;
	auto centerY = (object.top + object.bottom) / 2;

	auto objectWidth = object.right - object.left + 1;
	auto objectHeight = object.bottom - object.top + 1;

	auto surroundingBoxWidth = 3 * objectWidth;
	auto surroundingBoxHeight = 3 * objectHeight;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;

	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < width ? centerX + surroundingBoxWidth / 2 : width - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < height ? centerY + surroundingBoxHeight / 2 : height - 1;

	unsigned char avgValOfSurroundingBox;
	unsigned char avgValOfCurrentRect;
	Util::CalculateAverage(frameOnHost, FourLimits(boxLeftTopY, boxRightBottomY, boxLeftTopX, boxRightBottomX), avgValOfSurroundingBox, width);
	Util::CalculateAverage(frameOnHost, object, avgValOfCurrentRect, width);

	auto convexThresholdProportion = static_cast<double>(1 + convexPartition) / convexPartition;
	auto concaveThresholdPropotion = static_cast<double>(1 - concavePartition) / concavePartition;
	auto convexThreshold = avgValOfSurroundingBox * convexThresholdProportion;
	auto concaveThreshold = avgValOfSurroundingBox * concaveThresholdPropotion;

	if (abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < MinDiffOfConvextAndConcaveThreshold)
		return false;

	unsigned char centerValue = 0;
	Util::CalCulateCenterValue(frameOnHost, centerValue, width, centerX, centerY);

	if (avgValOfCurrentRect > convexThreshold || avgValOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

inline bool Filter::CheckOriginalImageSuroundedBox(unsigned char* originalFrameOnHost, int width, int height, const FourLimits& object)
{
	return CheckPeakValueAndAverageValue(originalFrameOnHost, width, height, object, ConvexPartitionOfOriginalImage, ConcavePartitionOfOriginalImage);
}

inline bool Filter::CheckDiscretizedImageSuroundedBox(unsigned char* discretizatedFrameOnHost, int width, int height, const FourLimits& object)
{
	return CheckPeakValueAndAverageValue(discretizatedFrameOnHost, width, height, object, ConvexPartitionOfDiscretizedImage, ConcavePartitionOfDiscretizedImage);
}

inline bool Filter::CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* frameOnHost, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckCoverageOfPreprocessedFrame(unsigned char* frameOnHost, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckInsideBoundaryDescendGradient(unsigned char* frameOnHost, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckStandardDeviation(unsigned char* frameOnHost, int width, const FourLimits& object)
{
	unsigned char averageValue;
	Util::CalculateAverage(frameOnHost, object, averageValue, width);

	uint64_t sum = 0;
	for (auto r = object.top; r <= object.bottom; ++r)
	{
		for (auto c = object.left; c <= object.right; ++c)
		{
			sum += (frameOnHost[r * width + c] - averageValue) * (frameOnHost[r * width + c] - averageValue);
		}
	}
	auto objectWidth = object.right - object.left + 1;
	auto objectHeight = object.bottom - object.top + 1;
	auto standardDeviation = sqrt(sum / (objectWidth * objectHeight));

	auto k = 2;
	auto adaptiveThreshold = standardDeviation * k + averageValue;

	if (adaptiveThreshold >= 150)
		return true;

	return false;
}
#endif
