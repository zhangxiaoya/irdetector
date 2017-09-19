#ifndef __FILTER_H__
#define __FILTER_H__
#include "../Models/FourLimits.h"

class Filter
{
public:
	static bool CheckOriginalImageSuroundedBox(unsigned char* frame, int width, const FourLimits& object);

	static bool CheckDiscretizedImageSuroundedBox(unsigned char* frame, int width, const FourLimits& object);

	static bool CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* frame, int width, const FourLimits& object);

	static bool CheckCoverageOfPreprocessedFrame(unsigned char* frame, int width, const FourLimits& object);

	static bool CheckInsideBoundaryDescendGradient(unsigned char* frame, int width, const FourLimits& object);

	static bool CheckStandardDeviation(unsigned char* frame, int width, const FourLimits& object);
};

inline bool Filter::CheckOriginalImageSuroundedBox(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckDiscretizedImageSuroundedBox(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckSurroundingBoundaryDiscontinuityAndDescendGradientOfPrerpocessedFrame(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckCoverageOfPreprocessedFrame(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckInsideBoundaryDescendGradient(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}

inline bool Filter::CheckStandardDeviation(unsigned char* frame, int width, const FourLimits& object)
{
	return true;
}
#endif
