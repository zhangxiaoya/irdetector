
#pragma once
struct TargetPosition
{
	int topLeftX;
	int topLeftY;
	int bottomRightX;
	int bottomRightY;
};

struct ResultSegment
{
	unsigned char header[16];
	int targetCount;
	TargetPosition targets[5];
};
