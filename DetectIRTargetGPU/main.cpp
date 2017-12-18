#include "Init/Init.hpp"
#include "Network/NetworkHelper.h"
#include "Validation/Validation.h"
#include "Headers/PreProcessParameters.h"

int main(int argc, char* argv[])
{
	// ��ʼ��CUDA�豸
	const auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		RunOnNetwork();

//		CheckConrrectness(IMAGE_WITDH, IMAGE_HEIGHT);

//		CheckPerformance(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);

//		CheckTracking(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);

//		CheckSearching(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);

		CheckLazyDetector(IMAGE_WIDTH, IMAGE_HEIGHT, DIALATION_KERNEL_RADIUS, DISCRETIZATION_SCALE);
	}

	// �ͷ�CUDA�豸
	CUDAInit::cudaDeviceRelease();

	// ϵͳ��ͣ
	system("Pause");
	return 0;
}
