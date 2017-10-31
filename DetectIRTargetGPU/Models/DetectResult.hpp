
#pragma once
#include "TargetPosition.hpp"

struct DetectResult
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
	bool hasTracker[5];
};
