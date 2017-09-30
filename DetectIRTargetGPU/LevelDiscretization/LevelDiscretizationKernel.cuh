#ifndef __LEVEL_DISCRETIZATION_KERNEL__
#define __LEVEL_DISCRETIZATION_KERNEL__

void LevelDiscretizationOnGPU(unsigned short* frameOnDevice, int width, int height, int discretizationScale);

#endif
