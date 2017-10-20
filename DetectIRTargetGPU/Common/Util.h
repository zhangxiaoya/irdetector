#ifndef __UTIL__
#define __UTIL__
#include "../Models/FourLimits.h"

class Util
{
public:
	static void CalculateAverage(unsigned short* frame, const FourLimits& object, unsigned short& averageValue, const int width);

	static void CalCulateCenterValue(unsigned short* discretization_result_on_host, unsigned short& center_value, int width, const int center_x, const int center_y);

	static void CalculateSurroundingValue(unsigned short* frame_of_original_image, const FourLimits& object, unsigned short& surrounding_average_value_of_origin_image, int width, int height);
};

inline void Util::CalculateAverage(unsigned short* frame, const FourLimits& object, unsigned short& averageValue, const int width)
{
	auto sum = 0;
	for (auto r = object.top; r <= object.bottom; ++r)
	{
		auto rowSum = 0;
		for(auto c = object.left; c <= object.right; ++c)
		{
			rowSum += static_cast<int>(frame[r * width + c]);
		}
		sum += static_cast<int>(rowSum / (object.right - object.left + 1));
	}
	averageValue = static_cast<unsigned short>(sum / (object.bottom - object.top + 1));
}

inline void Util::CalCulateCenterValue(unsigned short* discretizationResultOnHost, unsigned short& centerValue, int width, const int centerX, const int centerY)
{
	auto sum = 0;
	sum += discretizationResultOnHost[centerY * width + centerX];
	sum += discretizationResultOnHost[centerY * width + centerX + 1];
	sum += discretizationResultOnHost[(centerY +1)* width + centerX ];
	sum += discretizationResultOnHost[(centerY +1)* width + centerX + 1];

	centerValue = static_cast<unsigned short>(sum / 4);
}

inline void Util::CalculateSurroundingValue(unsigned short* frame_of_original_image, const FourLimits& object, unsigned short& surrounding_average_value_of_origin_image, int width, int height)
{
	auto objectWidth = object.right - object.left + 1;
	auto objectHeight = object.bottom - object.top + 1;

	auto surroundingBoxWidth = 3 * objectWidth;
	auto surroundingBoxHeight = 3 * objectHeight;

	auto centerX = (object.left + object.right) / 2;
	auto centerY = (object.bottom + object.top) / 2;

	auto boxLeftTopX = centerX - surroundingBoxWidth / 2 >= 0 ? centerX - surroundingBoxWidth / 2 : 0;
	auto boxLeftTopY = centerY - surroundingBoxHeight / 2 >= 0 ? centerY - surroundingBoxHeight / 2 : 0;

	auto boxRightBottomX = centerX + surroundingBoxWidth / 2 < width ? centerX + surroundingBoxWidth / 2 : width - 1;
	auto boxRightBottomY = centerY + surroundingBoxHeight / 2 < height ? centerY + surroundingBoxHeight / 2 : height - 1;

	CalculateAverage(frame_of_original_image, FourLimits(boxLeftTopY, boxRightBottomY, boxLeftTopX, boxRightBottomX), surrounding_average_value_of_origin_image, width);
}
#endif
