#pragma once
#include "TargetPosition.hpp"

struct ResultSegment
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
};
