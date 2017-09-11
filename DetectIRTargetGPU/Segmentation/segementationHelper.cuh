#include <host_defines.h>
#ifndef __SEGEMENTATION_HELPER_H__
#define __SEGEMENTATION_HELPER_H__

extern inline void Segmentation(unsigned char* frame, int width, int height);

extern __global__ void SplitByLevel(unsigned char* frame, unsigned char* dstFrame, int width, int height, unsigned char levelVal);

#endif
