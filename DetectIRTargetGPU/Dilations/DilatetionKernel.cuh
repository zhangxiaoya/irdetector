#ifndef __DIALATION_KERNEL__
#define __DIALATION_KERNEL__

void DilationFilter(unsigned short* srcFrameOnDevice, unsigned short* dstFrameOnDevice, int width, int height, int radius);

#endif
