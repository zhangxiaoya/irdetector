#include <host_defines.h>
#ifndef __MESH_KERNEL_D_H__
#define __MESH_KERNEL_D_H__

const auto BlockX = 32;
const auto BlockY = 32;

void MeshCCL(unsigned char* frame, int* label, int width, int height);

__device__ int IntMin(int a, int b);

__global__ void InitCCL(int* label, int* reference, int width, int height);

__global__ void MeshKernelDScanning(unsigned char* frame, int* label, int* reference, const int width, const int height, bool* iterationFlag);

__global__ void MeshKernelDAnalysis(int* label, int* reference, const int width, const int height);

__global__ void MeshKernelDLabelling(int* label, int* reference, const int width, const int height);

#endif
