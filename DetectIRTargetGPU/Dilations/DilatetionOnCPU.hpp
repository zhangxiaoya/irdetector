#ifndef __DILATIONONCPU__
#define __DILATIONONCPU__
#include <limits>
#include "../Common/Util.h"

class DilationOnCPU
{
public:
	static void erosionCPU(int* src, int* dst, int width, int height, int radio);
};

inline void DilationOnCPU::erosionCPU(int* src, int* dst, int width, int height, int radio)
{
	auto tmp = new int[width * height];
	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startCol = IMAX(0, c - radio);
			int endCol = IMIN(width - 1, c + radio);

			auto value = std::numeric_limits<int>::max();

			for (auto windowCol = startCol; windowCol <= endCol; windowCol++)
			{
				value = IMIN(src[r * width + windowCol], value);
			}
			tmp[r * width + c] = value;
		}
	}

	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startRow = IMAX(0, r - radio);
			int endRow = IMIN(height - 1, r + radio);

			auto value = std::numeric_limits<int>::max();

			for (auto windowRow = startRow; windowRow <= endRow; windowRow++)
			{
				value = IMIN(tmp[windowRow * width + c], value);
			}
			dst[r * width + c] = value;
		}
	}
	delete[] tmp;
}

#endif
