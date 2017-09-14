#pragma once
#include "Point.h"

struct ObjectRect
{
	ObjectRect() : width(0), height(0)
	{
	}

	Point lt;
	Point rb;
	int width;
	int height;
};
