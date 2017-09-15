#ifndef __DILATIONONCPU__
#define __DILATIONONCPU__
#include <limits>
#include "../Common/Util.h"

inline unsigned char UCMaxOnHost(unsigned char a, unsigned char b)
{
	return a > b ? a : b;
}

inline unsigned char UCMinOnHost(unsigned char a, unsigned char b)
{
	return a > b ? b : a;
}

class DilationOnCPU
{
public:
	static void ErosionCPU(unsigned char* srcFrameOnHost, unsigned char* dstFrameOnHost, int width, int height, int radius);

	static void DilationCPU(unsigned char* src, unsigned char* dst, int width, int height, int radio);
};

inline void DilationOnCPU::ErosionCPU(unsigned char* srcFrameOnHost, unsigned char* dstFrameOnHost, int width, int height, int radius)
{
	auto tmp = new int[width * height];
	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startCol = IMAX(0, c - radius);
			int endCol = IMIN(width - 1, c + radius);

			auto value = std::numeric_limits<int>::max();

			for (auto windowCol = startCol; windowCol <= endCol; windowCol++)
			{
				value = IMIN(srcFrameOnHost[r * width + windowCol], value);
			}
			tmp[r * width + c] = value;
		}
	}

	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startRow = IMAX(0, r - radius);
			int endRow = IMIN(height - 1, r + radius);

			auto value = std::numeric_limits<int>::max();

			for (auto windowRow = startRow; windowRow <= endRow; windowRow++)
			{
				value = IMIN(tmp[windowRow * width + c], value);
			}
			dstFrameOnHost[r * width + c] = value;
		}
	}
	delete[] tmp;
}

inline void DilationOnCPU::DilationCPU(unsigned char* srcFrameOnHost, unsigned char* dstFrameOnHost, int width, int height, int radius)
{
	auto tmp = new int[width * height];
	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startCol = IMAX(0, c - radius);
			int endCol = IMIN(width - 1, c + radius);

			auto value = std::numeric_limits<unsigned char>::min();

			for (auto windowCol = startCol; windowCol <= endCol; windowCol++)
			{
				value = UCMaxOnHost(srcFrameOnHost[r * width + windowCol], value);
			}
			tmp[r * width + c] = value;
		}
	}

	for (auto r = 0; r < height; r++)
	{
		for (auto c = 0; c < width; c++)
		{
			int startRow = IMAX(0, r - radius);
			int endRow = IMIN(height - 1, r + radius);

			auto value = std::numeric_limits<unsigned char>::min();

			for (auto windowRow = startRow; windowRow <= endRow; windowRow++)
			{
				value = UCMaxOnHost(tmp[windowRow * width + c], value);
			}
			dstFrameOnHost[r * width + c] = value;
		}
	}
	delete[] tmp;
}

#endif
