#ifndef __UTIL__
#define __UTIL__
#include "../Models/FourLimits.h"

class Util
{
public:
	static void CalculateAverage(unsigned short* frame, const FourLimits& object, unsigned short& averageValue, const int width);

	static void CalCulateCenterValue(unsigned short* discretization_result_on_host, unsigned short& center_value, int width, const int center_x, const int center_y);
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
#endif
