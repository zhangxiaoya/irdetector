#pragma once
struct TargetPosition
{
	TargetPosition() :
		topLeftY(-1),
		topLeftX(-1),
		bottomRightX(-1),
		bottomRightY(-1)
	{
	}

	int topLeftX;
	int topLeftY;
	int bottomRightX;
	int bottomRightY;
};
