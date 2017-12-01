#include "Init/Init.hpp"
#include "Network/NetworkHelper.h"
#include "Validation/Validation.h"
#include "Headers/PreProcessParameters.h"

int main(int argc, char* argv[])
{
	// 初始化CUDA设备
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		RunOnNetwork();

//		CheckConrrectness(IMAGE_WITDH, IMAGE_HEIGHT);

//		CheckPerformance(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

		CheckSearching(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);
	}

	// 释放CUDA设备
	CUDAInit::cudaDeviceRelease();

	// 系统暂停
	system("Pause");
	return 0;
}
