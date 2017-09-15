#ifndef __CHECK_DILATION__
#define __CHECK_DILATION__

#include <sstream>
#include "../Headers/GlobalMainHeaders.h"
#include "../Models/LogLevel.hpp"

class CheckDiff
{
public:
	template<typename T>
	static bool Check(T* resultOnCPU, T* resultOnGPU, int width, int height);
};

template<typename T>
bool CheckDiff::Check(T* resultOnCPU, T* resultOnGPU, int width, int height)
{
	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			if (resultOnCPU[r * width + c] != resultOnGPU[r * width + c])
			{
				std::ostringstream oss;
				oss << "Expected: " << static_cast<int>(resultOnCPU[r * width + c]) << ", actual: " << static_cast<int>(resultOnGPU[r * width + c]) << ", on: " << r << ", " << c << std::endl;
				auto errorMsg = oss.str();
				logPrinter.PrintLogs(errorMsg, Error);
				return false;
			}
		}
	}
	return true;
}

#endif
