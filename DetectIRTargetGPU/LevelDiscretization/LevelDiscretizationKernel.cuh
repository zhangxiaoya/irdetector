#ifndef __LEVEL_DISCRETIZATION_KERNEL__
#define __LEVEL_DISCRETIZATION_KERNEL__

void LevelDiscretizationOnGPU(unsigned char* frameOnDevice, int width, int height, int discretizationScale);

#endif
