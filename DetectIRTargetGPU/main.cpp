#include "Assistants/ShowFrame.hpp"
#include "Validation/Validation.hpp"
#include "Init/Init.hpp"

const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 2;

int main(int argc, char* argv[])
{
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		Validation validation;
		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		validation.VailidationAll();

		//			logPrinter.PrintLogs("segementation On GPU", Info);
		//			Segmentation(dilationResultOfGPU, WIDTH, HEIGHT);
	}
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
