#pragma once
#include "TargetPosition.hpp"

struct DetectResultSegment
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
};


struct DetectResultWithTrackerStatus
{
	DetectResultSegment* detectResultPointers;
	bool hasTracker[5];
};