#ifndef __DIALATION_KERNEL__
#define __DIALATION_KERNEL__

void DilationFilter(unsigned char* srcFrameOnDevice, unsigned char* dstFrameOnDevice, int width, int height, int radius);

#endif
