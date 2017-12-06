#pragma
#include "TargetPosition.hpp"

struct TraceItem
{
	TraceItem():FrameIndex(-1)
	{
	}

	int FrameIndex;
	TargetPosition Pos;
};