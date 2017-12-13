#pragma once
#include "TargetPosition.hpp"

struct FourLimits
{
	explicit FourLimits(int _top = -1, int _bottom = -1, int _left = -1, int _right = -1)
		: top(_top),
		  bottom(_bottom),
		  left(_left),
		  right(_right)
	{
	}

	FourLimits(TargetPosition& pos)
	{
		this->top = pos.topLeftY;
		this->left = pos.topLeftX;
		this->right = pos.bottomRightX;
		this->bottom = pos.bottomRightY;
	}

	int top;
	int bottom;
	int left;
	int right;
};