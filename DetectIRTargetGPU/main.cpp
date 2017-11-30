#include "Init/Init.hpp"
#include "Network/NetworkHelper.h"

int main(int argc, char* argv[])
{
	// 初始化CUDA设备
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		RunOnNetwork();

//		CheckConrrectness(IMAGE_WITDH, IMAGE_HEIGHT);

//		CheckPerformance(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

//		CheckSearching(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);
	}

	// 释放CUDA设备
	CUDAInit::cudaDeviceRelease();

	// 系统暂停
	system("Pause");
	return 0;
}
