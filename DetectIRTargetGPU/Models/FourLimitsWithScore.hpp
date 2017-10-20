#pragma once
#include "../Models/FourLimits.h"

class FourLimitsWithScore
{
public:
	FourLimitsWithScore(): score(0.0)
	{
	}

	FourLimits object;
	double score;
};
