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

	static bool CheckDiscretizedImageSuroundedBox(unsigned char* preprocessedFrameOnHost, int width, int height, const FourLimits& object);

	static bool CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* preprocessedFrameOnHost, int width, int height, const FourLimits& object);

	static bool CheckCoverageOfPreprocessedFrame(unsigned char* preprocessedFrameOnHost, int width, const FourLimits& object);

	static bool CheckInsideBoundaryDescendGradient(unsigned char* originalFrameOnHost, int width, const FourLimits& object);

	static bool CheckStandardDeviation(unsigned char* originalFrameOnHost, int width, const FourLimits& object);

	static void InitBuObject(unsigned char* frameOfOriginalImage, unsigned char* frameOfPreprocessedImage, const FourLimits& object, int width);

private:
	static bool CheckPeakValueAndAverageValue(unsigned char* frameOnHost, int width, int height, const FourLimits& object, unsigned char centerValueOfCurrentRect, int convexPartition, int concavePartition);

	static int centerX;
	static int centerY;
	static int objectWidth;
	static int objectHeight;
	static unsigned char centerValueOfOriginalImage;
	static unsigned char centerValueOfPreprocessedImage;
	static unsigned char averageValueOfOriginalImage;
	static unsigned char averageValueOfPreprocessedImage;
};

inline bool Filter::CheckPeakValueAndAverageValue(unsigned char* frameOnHost, int width, int height, const FourLimits& object, unsigned char centerValueOfCurrentRect, int convexPartition, int concavePartition)
{
	auto surroundingBoxWidth = 3 * objectWidth;
	auto surroundingBoxHeight = 3 * objectHeight;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;

	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < width ? centerX + surroundingBoxWidth / 2 : width - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < height ? centerY + surroundingBoxHeight / 2 : height - 1;

	unsigned char avgValOfSurroundingBox;
	Util::CalculateAverage(frameOnHost, FourLimits(boxLeftTopY, boxRightBottomY, boxLeftTopX, boxRightBottomX), avgValOfSurroundingBox, width);

	auto convexThresholdProportion = static_cast<double>(1 + convexPartition) / convexPartition;
	auto concaveThresholdPropotion = static_cast<double>(1 - concavePartition) / concavePartition;
	auto convexThreshold = avgValOfSurroundingBox * convexThresholdProportion;
	auto concaveThreshold = avgValOfSurroundingBox * concaveThresholdPropotion;

	if (abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < MinDiffOfConvextAndConcaveThreshold)
		return false;

	unsigned char centerValue = 0;
	Util::CalCulateCenterValue(frameOnHost, centerValue, width, centerX, centerY);

	if (centerValueOfCurrentRect > convexThreshold || centerValueOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

inline bool Filter::CheckOriginalImageSuroundedBox(unsigned char* originalFrameOnHost, int width, int height, const FourLimits& object)
{
	return CheckPeakValueAndAverageValue(originalFrameOnHost, width, height, object, centerValueOfOriginalImage, ConvexPartitionOfOriginalImage, ConcavePartitionOfOriginalImage);
}

inline bool Filter::CheckDiscretizedImageSuroundedBox(unsigned char* preprocessedFrameOnHost, int width, int height, const FourLimits& object)
{
	return CheckPeakValueAndAverageValue(preprocessedFrameOnHost, width, height, object, centerValueOfPreprocessedImage, ConvexPartitionOfDiscretizedImage, ConcavePartitionOfDiscretizedImage);
}

inline bool Filter::CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* preprocessedFrameOnHost, int width, int height, const FourLimits& object)
{
	auto pixelValueOverCenterValueCount = 0;
	auto pixelCountOfSurroundingBoundary = 0;

	auto sum = 0;
	auto topRow = object.top - 1;
	if (topRow >= 0)
	{
		for (auto c = object.left; c <= object.right; ++c)
		{
			auto val = preprocessedFrameOnHost[topRow * width + c];
			if (val > centerValueOfPreprocessedImage)
				pixelValueOverCenterValueCount++;
			sum += static_cast<int>(val);
		}
		pixelCountOfSurroundingBoundary += (object.right - object.left + 1);
	}

	auto bottomRow = object.bottom + 1;
	if (bottomRow < height)
	{
		for (auto c = object.left; c <= object.right; ++c)
		{
			auto val = preprocessedFrameOnHost[bottomRow * width + c];
			if (val > centerValueOfPreprocessedImage)
				pixelValueOverCenterValueCount++;
			sum += static_cast<int>(val);
		}
		pixelCountOfSurroundingBoundary += (object.right - object.left + 1);
	}

	auto leftCol = object.left - 1;
	if (leftCol >= 0)
	{
		for (auto r = object.top; r <= object.bottom; ++r)
		{
			auto val = preprocessedFrameOnHost[r * width + leftCol];
			if (val > centerValueOfPreprocessedImage)
				pixelValueOverCenterValueCount++;
			sum += static_cast<int>(val);
		}
		pixelCountOfSurroundingBoundary += (object.bottom - object.top + 1);
	}

	auto rightCol = object.right + 1;
	if (rightCol < width)
	{
		for (auto r = object.top; r <= object.bottom; ++r)
		{
			auto val = preprocessedFrameOnHost[r * width + rightCol];
			if (val > centerValueOfPreprocessedImage)
				pixelValueOverCenterValueCount++;
			sum += static_cast<int>(val);
		}
		pixelCountOfSurroundingBoundary += (object.bottom - object.top + 1);
	}

	if (leftCol >= 0 && topRow >= 0)
	{
		auto val = preprocessedFrameOnHost[topRow * width + leftCol];
		if (val > centerValueOfPreprocessedImage)
			pixelValueOverCenterValueCount++;
		sum += static_cast<int>(val);
		pixelCountOfSurroundingBoundary++;
	}
	if (leftCol >= 0 && bottomRow < height)
	{
		auto val = preprocessedFrameOnHost[bottomRow * width + leftCol];
		if (val > centerValueOfPreprocessedImage)
			pixelValueOverCenterValueCount++;
		sum += static_cast<int>(val);
		pixelCountOfSurroundingBoundary++;
	}
	if (rightCol < width && topRow >= 0)
	{
		auto val = preprocessedFrameOnHost[topRow * width + rightCol];
		if (val > centerValueOfPreprocessedImage)
			pixelValueOverCenterValueCount++;
		sum += static_cast<int>(val);
		pixelCountOfSurroundingBoundary++;
	}
	if (rightCol < width && bottomRow < height)
	{
		auto val = preprocessedFrameOnHost[bottomRow * width + rightCol];
		if (val > centerValueOfPreprocessedImage)
			pixelValueOverCenterValueCount++;
		sum += static_cast<int>(val);
		pixelCountOfSurroundingBoundary++;
	}

	auto avgSurroundingPixels = static_cast<unsigned char>(sum / pixelCountOfSurroundingBoundary);

	if (pixelValueOverCenterValueCount < 2 && avgSurroundingPixels < (averageValueOfPreprocessedImage * 11 / 12))
		return true;

	return false;
}

inline bool Filter::CheckCoverageOfPreprocessedFrame(unsigned char* preprocessedFrameOnHost, int width, const FourLimits& object)
{
	auto sum = 0;
	sum += preprocessedFrameOnHost[object.top * width + object.left];
	sum += preprocessedFrameOnHost[object.bottom * width + object.left];
	sum += preprocessedFrameOnHost[object.top * width + object.right];
	sum += preprocessedFrameOnHost[object.bottom * width + object.right];

	auto maxValue = static_cast<unsigned char>(sum / 4);

	auto count = 0;
	for (auto r = object.top; r <= object.bottom; ++r)
	{
		for (auto c = object.left; c <= object.right; ++c)
		{
			if (preprocessedFrameOnHost[r * width + c] >= maxValue)
				count++;
		}
	}

	if (static_cast<double>(count) / (objectHeight * objectWidth) > 0.15)
		return true;

	return false;
}

inline bool Filter::CheckInsideBoundaryDescendGradient(unsigned char* originalFrameOnHost, int width, const FourLimits& object)
{
	auto topRowSum = 0;
	auto bottomRowSum = 0;
	for (auto c = object.left; c <= object.right; ++c)
	{
		topRowSum += static_cast<int>(originalFrameOnHost[object.top * width + c]);
		bottomRowSum += static_cast<int>(originalFrameOnHost[object.bottom * width + c]);
	}
	auto avgTop = static_cast<unsigned char>(topRowSum / (object.right - object.left + 1));
	auto avgBottom = static_cast<unsigned char>(bottomRowSum / (object.right - object.left + 1));

	auto leftSum = 0;
	auto rightSum = 0;
	for (auto r = object.top; r <= object.bottom; ++r)
	{
		leftSum += static_cast<int>(originalFrameOnHost[r * width + object.left]);
		rightSum += static_cast<int>(originalFrameOnHost[r * width + object.right]);
	}
	auto avgLeft = static_cast<int>(leftSum / (object.bottom - object.top + 1));
	auto avgRight = static_cast<int>(rightSum / (object.bottom - object.top + 1));

	auto count = 0;
	if (avgLeft < averageValueOfOriginalImage) count++;
	if (avgBottom < averageValueOfOriginalImage) count++;
	if (avgRight < averageValueOfOriginalImage) count++;
	if (avgTop < averageValueOfOriginalImage) count++;

	if (count > 3)
		return true;
	return false;
}

inline bool Filter::CheckStandardDeviation(unsigned char* originalFrameOnHost, int width, const FourLimits& object)
{
	uint64_t sum = 0;
	for (auto r = object.top; r <= object.bottom; ++r)
	{
		for (auto c = object.left; c <= object.right; ++c)
		{
			sum += (originalFrameOnHost[r * width + c] - averageValueOfOriginalImage) * (originalFrameOnHost[r * width + c] - averageValueOfOriginalImage);
		}
	}
	auto objectWidth = object.right - object.left + 1;
	auto objectHeight = object.bottom - object.top + 1;
	auto standardDeviation = sqrt(sum / (objectWidth * objectHeight));

	auto k = 2;
	auto adaptiveThreshold = standardDeviation * k + averageValueOfOriginalImage;

	if (adaptiveThreshold >= 150)
		return true;

	return false;
}

inline void Filter::InitBuObject(unsigned char* frameOfOriginalImage, unsigned char* frameOfPreprocessedImage, const FourLimits& object, int width)
{
	centerX = (object.left + object.right) / 2;
	centerY = (object.top + object.bottom) / 2;

	objectWidth = object.right - object.left + 1;
	objectHeight = object.bottom - object.top + 1;

	Util::CalculateAverage(frameOfOriginalImage, object, averageValueOfOriginalImage, width);
	Util::CalculateAverage(frameOfPreprocessedImage, object, averageValueOfPreprocessedImage, width);
	Util::CalCulateCenterValue(frameOfOriginalImage, centerValueOfOriginalImage, width, centerX, centerY);
	Util::CalCulateCenterValue(frameOfPreprocessedImage, centerValueOfPreprocessedImage, width, centerX, centerY);
}

#endif
