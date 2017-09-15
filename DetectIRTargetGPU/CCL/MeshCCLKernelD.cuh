#ifndef __MESH_KERNEL_D_H__
#define __MESH_KERNEL_D_H__

const auto BlockX = 32;
const auto BlockY = 32;

void MeshCCL(unsigned char* frameOnDevice, int* labelsOnDevice, int width, int height);

#endif
