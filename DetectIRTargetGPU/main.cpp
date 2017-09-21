#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 2;

int main(int argc, char* argv[])
{
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
		DataReceiver::InitNetworks();
		DataReceiver::Run();
		DataReceiver::DestroyNetWork();


		Validation validation;
		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		validation.VailidationAll();

		DetectorValidation visualEffectValidator;
		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		visualEffectValidator.VailidationAll();
	}
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
