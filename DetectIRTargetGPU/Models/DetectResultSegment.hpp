#pragma once
#include "TargetPosition.hpp"
#include "../Headers/DetectorParameters.h"

struct DetectResultSegment
{
	unsigned char header[FRAME_HEADER_LENGTH];
	int targetCount;
	TargetPosition targets[MAX_DETECTED_TARGET_COUNT];
};


struct DetectResultWithTrackerStatus
{
	DetectResultSegment* detectResultPointers;
	bool hasTracker[MAX_DETECTED_TARGET_COUNT];
};