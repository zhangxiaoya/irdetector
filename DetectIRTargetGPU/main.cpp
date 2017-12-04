#include "Init/Init.hpp"
#include "Network/NetworkHelper.h"

int main(int argc, char* argv[])
{
	// ��ʼ��CUDA�豸
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		RunOnNetwork();

//		CheckConrrectness(IMAGE_WITDH, IMAGE_HEIGHT);

//		CheckPerformance(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

//		CheckTracking(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);

//		CheckSearching(IMAGE_WIDTH, IMAGE_HEIGHT, DilationRadius, DiscretizationScale);
	}

	// �ͷ�CUDA�豸
	CUDAInit::cudaDeviceRelease();

	// ϵͳ��ͣ
	system("Pause");
	return 0;
}
