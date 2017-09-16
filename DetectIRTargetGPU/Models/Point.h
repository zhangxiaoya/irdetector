#pragma once
struct Point
{
	explicit Point(int _x = -1, int _y = -1) : x(_x), y(_y)
	{
	}

	int x;
	int y;
};