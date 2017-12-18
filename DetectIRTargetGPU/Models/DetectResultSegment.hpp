#pragma once
#include "../Headers/DetectorParameters.h"

#ifndef TARGET_POSTION
#define TARGET_POSTION

struct TargetPosition
{
	TargetPosition() :
		topLeftY(0),
		topLeftX(0),
		bottomRightX(0),
		bottomRightY(0)
	{
	}

	unsigned short topLeftX;
	unsigned short topLeftY;
	unsigned short bottomRightX;
	unsigned short bottomRightY;
};

#endif // !TARGET_POSTION

#ifndef TARGET_INFO
#define TARGET_INFO

struct TargetInfo
{
	TargetInfo() :
		avgValue(0),
		placeHolder_1(0),
		placeHolder_2(0),
		placeHolder_3(0)
	{
	}

	unsigned short avgValue;
	unsigned short placeHolder_1;
	unsigned short placeHolder_2;
	unsigned short placeHolder_3;
};

#endif // !TARGET_INFO

#ifndef DETECT_RESULT_SEGMENT
#define DETECT_RESULT_SEGMENT

struct DetectResultSegment
{
	unsigned char header[FRAME_HEADER_LENGTH];
	unsigned short targetCount;
	TargetPosition targets[MAX_DETECTED_TARGET_COUNT];
	TargetInfo targetInfo[MAX_DETECTED_TARGET_COUNT];
};

#endif // !DETECT_RESULT_SEGMENT

#ifndef DETECT_RESULT_WITH_TRACKER_STATUS
#define DETECT_RESULT_WITH_TRACKER_STATUS

struct DetectResultWithTrackerStatus
{
	DetectResultSegment* detectResultPointers;
	bool hasTracker[MAX_DETECTED_TARGET_COUNT];
};

#endif // !DETECT_RESULT_WITH_TRACKER_STATUS