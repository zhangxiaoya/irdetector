#ifndef __UTIL__
#define __UTIL__
#include "../Models/FourLimits.h"

class Util
{
public:
	static unsigned char CalculateCenterValue(unsigned char* frame, const FourLimits& object, const int width);
};

inline unsigned char Util::CalculateCenterValue(unsigned char* frame, const FourLimits& object, const int width)
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
	return static_cast<unsigned char>(sum / (object.bottom - object.top));
}
#endif
