#include "Validation/Validation.hpp"
#include "Init/Init.hpp"
#include "Validation/DetectorValidation.hpp"
#include "Network/DataReceiver.h"

const unsigned int WIDTH = 320;
const unsigned int HEIGHT = 256;
const unsigned BYTESIZE = 1;

unsigned char FrameData[WIDTH * HEIGHT * BYTESIZE];

Detector* detector = new Detector();

HANDLE dataMutex;

DWORD WINAPI Detect(LPVOID lpParam)
{
	WaitForSingleObject(dataMutex, INFINITE);
	CheckPerf(detector->DetectTargets(FrameData), "whole");
	ReleaseMutex(dataMutex);
	return 0;
}

DWORD WINAPI ReadData(LPVOID lpParam)
{
	WaitForSingleObject(dataMutex, INFINITE);
	Run(FrameData);
	ReleaseMutex(dataMutex);

	return 0;
}


int main(int argc, char* argv[])
{
	// Init CUDA
	auto cudaInitStatus = CUDAInit::cudaDeviceInit();
	if (cudaInitStatus)
	{
//		// Create Mutex
//		dataMutex = CreateMutex(nullptr, FALSE, nullptr);
//
//		// Init Network socket
//		InitNetworks();
//
//		// Init Detector Space
//		detector->InitSpace();
//
//		HANDLE hThread1;
//		HANDLE hThread2;
//		while (true)
//		{
//			hThread1 = CreateThread(nullptr,
//			                        0,
//			                        ReadData,
//			                        nullptr,
//			                        0,
//			                        nullptr);
//			hThread2 = CreateThread(nullptr,
//			                        0,
//			                        Detect,
//			                        nullptr,
//			                        0,
//			                        nullptr);
//		}
//
//		CloseHandle(hThread1);
//		CloseHandle(hThread2);
//		DestroyNetWork();
//
//
////		Validation validation;
////		validation.InitValidationData("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
////		validation.VailidationAll();

		DetectorValidation visualEffectValidator;
//		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin");
		visualEffectValidator.InitDataReader("D:\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_2.bin");
		visualEffectValidator.VailidationAll();
	}
	CUDAInit::cudaDeviceRelease();

	system("Pause");
	return 0;
}
