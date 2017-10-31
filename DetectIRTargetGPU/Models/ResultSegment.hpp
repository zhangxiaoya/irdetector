#pragma once
#include "TargetPosition.hpp"

struct ResultSegment
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
};


struct DetectResult
{
	ResultSegment* result;
	bool hasTracker[5];
};