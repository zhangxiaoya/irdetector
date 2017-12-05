#pragma once
#include "TargetPosition.hpp"
#include "../Headers/DetectorParameters.h"

struct DetectResultSegment
{
	unsigned char header[FRAME_HEADER_LENGTH];
	int targetCount;
	TargetPosition targets[5];
};


struct DetectResultWithTrackerStatus
{
	DetectResultSegment* detectResultPointers;
	bool hasTracker[5];
};