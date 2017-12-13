#ifndef _DETECTED_TARGET_LAZY_CHECKER
#define _DETECTED_TARGET_LAZY_CHECKER

#include "FourLimits.h"

struct DetectedTargetLazyChecker
{
	DetectedTargetLazyChecker() :position(FourLimits()), lifeTime(0), count(0)
	{
	}
	FourLimits position;
	short lifeTime;
	short count;
};
#endif // !_DETECTED_TARGET_LAZY_CHECKER
