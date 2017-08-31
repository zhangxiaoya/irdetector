#ifndef __CHECK_DILATION__
#define __CHECK_DILATION__

#include <sstream>
#include "../Headers/GlobalMainHeaders.h"
#include "../Models/LogLevel.hpp"

class CheckDilation
{
public:
	static bool CheckDiff(unsigned char* resultOnCPU, unsigned char* resultOnGPU, int width, int height);
};

inline bool CheckDilation::CheckDiff(unsigned char* resultOnCPU, unsigned char* resultOnGPU, int width, int height)
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
				logPrinter.PrintLogs(errorMsg, LogLevel::Error);
				return false;
			}
		}
	}
	return true;
}

#endif
