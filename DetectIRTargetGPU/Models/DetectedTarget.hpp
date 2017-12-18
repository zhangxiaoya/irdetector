#pragma once
#include "FourLimits.h"

struct DetectedTarget
{
	DetectedTarget():
		fourLimits(FourLimits()),
		width(-1),
		height(-1),
		centerX(-1),
		centerY(-1)
	{
	}

	FourLimits fourLimits;

	int width;
	int height;
	int centerX;
	int centerY;
};
