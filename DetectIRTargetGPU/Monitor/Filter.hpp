#ifndef __FILTER_H__
#define __FILTER_H__

#include "../Models/FourLimits.h"
#include "../Common/Util.h"
#include <cmath>
#include "../Headers/GlobalMainHeaders.h"

class Filter
{
public:
	Filter();

	bool CheckOriginalImageSuroundedBox(unsigned short* originalFrameOnHost, int width, int height, const FourLimits& object) const;

	bool CheckDiscretizedImageSuroundedBox(unsigned short* preprocessedFrameOnHost, int width, int height, const FourLimits& object) const;

	bool CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned short* preprocessedFrameOnHost, int width, int height, const FourLimits& object) const;

	bool CheckCoverageOfPreprocessedFrame(unsigned short* preprocessedFrameOnHost, int width, const FourLimits& object) const;

	bool CheckInsideBoundaryDescendGradient(unsigned short* originalFrameOnHost, int width, const FourLimits& object) const;

	bool CheckStandardDeviation(unsigned short* originalFrameOnHost, int width, const FourLimits& object) const;

	void InitObjectParameters(unsigned short* frameOfOriginalImage, unsigned short* frameOfPreprocessedImage, const FourLimits& object, int width);

public:
	void SetConvexPartitionOfOriginalImage(int value);

	void SetConcavePartitionOfOriginalImage(int value);

	void SetConvexPartitionOfDiscretizedImage(int value);

	void SetConcavePartitionOfDiscretizedImage(int value);

	unsigned short GetCenterValue() const
	{
		return this->centerValueOfPreprocessedImage;
	}

private:
	bool CheckPeakValueAndAverageValue(unsigned short* frameOnHost,
	                                   int width,
	                                   int height,
	                                   const FourLimits& object,
	                                   unsigned short centerValueOfCurrentRect,
	                                   int convexPartition,
	                                   int concavePartition) const;

	int centerX;
	int centerY;
	int objectWidth;
	int objectHeight;
	unsigned short centerValueOfOriginalImage;
	unsigned short centerValueOfPreprocessedImage;
	unsigned short averageValueOfOriginalImage;
	unsigned short averageValueOfPreprocessedImage;

	int ConvexPartitionOfOriginalImage;
	int ConcavePartitionOfOriginalImage;
	int ConvexPartitionOfDiscretizedImage;
	int ConcavePartitionOfDiscretizedImage;

	int const MinDiffOfConvextAndConcaveThreshold;
};

inline bool Filter::CheckPeakValueAndAverageValue(unsigned short* frameOnHost, int width, int height, const FourLimits& object, unsigned short centerValueOfCurrentRect, int convexPartition, int concavePartition) const
{
	auto surroundingBoxWidth = 3 * objectWidth;
	auto surroundingBoxHeight = 3 * objectHeight;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;

	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < width ? centerX + surroundingBoxWidth / 2 : width - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < height ? centerY + surroundingBoxHeight / 2 : height - 1;

	unsigned short avgValOfSurroundingBox;
	Util::CalculateAverage(frameOnHost, FourLimits(boxLeftTopY, boxRightBottomY, boxLeftTopX, boxRightBottomX), avgValOfSurroundingBox, width);

	auto convexThresholdProportion = static_cast<double>(1 + convexPartition) / convexPartition;
	auto concaveThresholdPropotion = static_cast<double>(1 - concavePartition) / concavePartition;
	auto convexThreshold = avgValOfSurroundingBox * convexThresholdProportion;
	auto concaveThreshold = avgValOfSurroundingBox * concaveThresholdPropotion;

	if (abs(static_cast<int>(convexThreshold) - static_cast<int>(concaveThreshold)) < MinDiffOfConvextAndConcaveThreshold)
		return false;

	unsigned short centerValue = 0;
	Util::CalCulateCenterValue(frameOnHost, centerValue, width, centerX, centerY);

	if (centerValueOfCurrentRect > convexThreshold || centerValueOfCurrentRect < concaveThreshold || centerValue > convexThreshold || centerValue < concaveThreshold)
	{
		return true;
	}
	return false;
}

inline Filter::Filter(): centerX(0), centerY(0),
                         objectWidth(0), objectHeight(0),
                         centerValueOfOriginalImage(0),
                         centerValueOfPreprocessedImage(0),
                         averageValueOfOriginalImage(0),
                         averageValueOfPreprocessedImage(0),
                         ConvexPartitionOfOriginalImage(0),
                         ConcavePartitionOfOriginalImage(0),
                         ConvexPartitionOfDiscretizedImage(0),
                         ConcavePartitionOfDiscretizedImage(0),
                         MinDiffOfConvextAndConcaveThreshold(3)
{
}

inline bool Filter::CheckOriginalImageSuroundedBox(unsigned short* originalFrameOnHost, int width, int height, const FourLimits& object) const
{
	return CheckPeakValueAndAverageValue(originalFrameOnHost, width, height, object, centerValueOfOriginalImage, ConvexPartitionOfOriginalImage, ConcavePartitionOfOriginalImage);
}

inline bool Filter::CheckDiscretizedImageSuroundedBox(unsigned short* preprocessedFrameOnHost, int width, int height, const FourLimits& object) const
{
	return CheckPeakValueAndAverageValue(preprocessedFrameOnHost, width, height, object, centerValueOfPreprocessedImage, ConvexPartitionOfDiscretizedImage, ConcavePartitionOfDiscretizedImage);
}

inline bool Filter::CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned short* preprocessedFrameOnHost, int width, int height, const FourLimits& object) const
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

	auto avgSurroundingPixels = static_cast<unsigned short>(sum / pixelCountOfSurroundingBoundary);

	if (pixelValueOverCenterValueCount < 2 && avgSurroundingPixels < (averageValueOfPreprocessedImage * 11 / 12))
		return true;

	return false;
}

inline bool Filter::CheckCoverageOfPreprocessedFrame(unsigned short* preprocessedFrameOnHost, int width, const FourLimits& object) const
{
	auto sum = 0;
	sum += preprocessedFrameOnHost[object.top * width + object.left];
	sum += preprocessedFrameOnHost[object.bottom * width + object.left];
	sum += preprocessedFrameOnHost[object.top * width + object.right];
	sum += preprocessedFrameOnHost[object.bottom * width + object.right];

	auto maxValue = static_cast<unsigned short>(sum / 4);

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

inline bool Filter::CheckInsideBoundaryDescendGradient(unsigned short* originalFrameOnHost, int width, const FourLimits& object) const
{
	auto topRowSum = 0;
	auto bottomRowSum = 0;
	for (auto c = object.left; c <= object.right; ++c)
	{
		topRowSum += static_cast<int>(originalFrameOnHost[object.top * width + c]);
		bottomRowSum += static_cast<int>(originalFrameOnHost[object.bottom * width + c]);
	}
	auto avgTop = static_cast<unsigned short>(topRowSum / (object.right - object.left + 1));
	auto avgBottom = static_cast<unsigned short>(bottomRowSum / (object.right - object.left + 1));

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

inline bool Filter::CheckStandardDeviation(unsigned short* originalFrameOnHost, int width, const FourLimits& object) const
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

inline void Filter::InitObjectParameters(unsigned short* frameOfOriginalImage, unsigned short* frameOfPreprocessedImage, const FourLimits& object, int width)
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

inline void Filter::SetConvexPartitionOfOriginalImage(int value)
{
	this->ConvexPartitionOfOriginalImage = value;
}

inline void Filter::SetConcavePartitionOfOriginalImage(int value)
{
	this->ConcavePartitionOfOriginalImage = value;
}

inline void Filter::SetConvexPartitionOfDiscretizedImage(int value)
{
	this->ConvexPartitionOfDiscretizedImage = value;
}

inline void Filter::SetConcavePartitionOfDiscretizedImage(int value)
{
	this->ConcavePartitionOfDiscretizedImage = value;
}

#endif
