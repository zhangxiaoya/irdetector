#pragma once
#include "DetectResultSegment.hpp"

struct FourLimits
{
	explicit FourLimits(int _top = -1, int _bottom = -1, int _left = -1, int _right = -1)
		: top(_top),
		  bottom(_bottom),
		  left(_left),
		  right(_right),
		  label(-1),
		  area(0)
	{
	}

	FourLimits(TargetPosition& pos)
	{
		this->top =	   (int)pos.topLeftY;
		this->left =   (int)pos.topLeftX;
		this->right =  (int)pos.bottomRightX;
		this->bottom = (int)pos.bottomRightY;
	}

	int top;
	int bottom;
	int left;
	int right;
	int label;
	int area;
};