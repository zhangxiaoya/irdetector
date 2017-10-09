#ifndef __SEGEMENTATION_HELPER_H__
#define __SEGEMENTATION_HELPER_H__
#include "../Models/FourLimits.h"
#include "../Models/ObjectRect.h"

class OverSegmentation
{
public:
	static inline void Segmentation(unsigned short* frameOnHost, int width, int height);

private:
	static void GenerateRect(int width, int height, FourLimits* allObjects, ObjectRect* allObjectRects);

	static void GetAllObjects(int width, int height, int* labelsOnHost, FourLimits* allObjects);
};

#endif
