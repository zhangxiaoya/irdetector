#pragma once
#include "../Models/FourLimits.h"

class FourLimitsWithScore
{
public:
	FourLimitsWithScore(): score(0)
	{
	}

	FourLimits object;
	int score;
};
