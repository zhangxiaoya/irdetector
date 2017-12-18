#pragma
#include "DetectResultSegment.hpp"

struct TraceItem
{
	TraceItem():FrameIndex(-1)
	{
	}

	int FrameIndex;
	TargetPosition Pos;
};