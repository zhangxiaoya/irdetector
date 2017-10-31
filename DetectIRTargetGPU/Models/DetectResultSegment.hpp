#pragma once
#include "TargetPosition.hpp"

struct DetectResultSegment
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
};


struct DetectResult
{
	DetectResultSegment* result;
	bool hasTracker[5];
};