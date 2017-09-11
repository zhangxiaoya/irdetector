#include <iostream>

#include "Headers/GlobalMainHeaders.h"
#include "DataReaderFromeFiles/BinaryFileReader.hpp"
#include "Dilations/DilatetionOnCPU.hpp"
#include "Dilations/DilatetionKernel.h"
#include "Checkers/CheckDiff.hpp"
#include "LevelDiscretization/LevelDiscretizationOnCPU.hpp"
#include "LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "Segmentation/segementationHelper.cuh"

inline bool cudaDeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		logPrinter.PrintLogs("CUDA error: no devices supporting CUDA.", Error);
		exit(EXIT_FAILURE);
	}

	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		logPrinter.PrintLogs("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", Error);
		return false;
	}
	logPrinter.PrintLogs("cudaSetDevice success!", Info);
	return true;
}

void cudaDeviceRelease()
{
	auto cudaResetStatus = cudaDeviceReset();
	if (cudaResetStatus == cudaSuccess)
	{
		logPrinter.PrintLogs("cudaDeviceReset success!", Info);
	}
	else
	{
		logPrinter.PrintLogs("cudaDeviceReset failed!", Waring);
	}
}

int main(int argc, char* argv[])
{
	auto cudaInitStatus = cudaDeviceInit(argc, const_cast<const char **>(argv));
	if (cudaInitStatus)
	{
		unsigned char* originalFrameOnDeivce;
		unsigned char* resultOnDevice;

		cudaMalloc(&originalFrameOnDeivce, WIDTH * HEIGHT);
		cudaMalloc(&resultOnDevice, WIDTH * HEIGHT);

		auto dilationResultOfCPU = new unsigned char[WIDTH * HEIGHT];
		auto dilationResultOfGPU = new unsigned char[WIDTH * HEIGHT];

		std::string fileName = "C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin";

		auto fileReader = new BinaryFileReader(fileName);

		fileReader->ReadBinaryFileToHostMemory();
		auto frameCount = fileReader->GetFrameCount();
		auto dataPoint = fileReader->GetDataPoint();

		auto iterationText = new char[200];
		for (auto i = 0; i < frameCount; ++i)
		{
			memset(dilationResultOfGPU, 0, WIDTH * HEIGHT);
			memset(dilationResultOfCPU, 0, WIDTH * HEIGHT);

			auto perFrame = dataPoint[i];

			logPrinter.PrintLogs("Dilation on CPU!", Info);
			DilationOnCPU::dilationCPU(perFrame, dilationResultOfCPU, WIDTH, HEIGHT, 1);

			sprintf_s(iterationText, 200, "Copy the %04d frame to device", i);
			logPrinter.PrintLogs(iterationText, Info);
			cudaMemcpy(originalFrameOnDeivce, perFrame, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

			logPrinter.PrintLogs("Dialtion On GPU", Info);
			FilterDilation(originalFrameOnDeivce, resultOnDevice, WIDTH, HEIGHT, 1);

			cudaMemcpy(dilationResultOfGPU, resultOnDevice, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

			if (!CheckDiff::Check(dilationResultOfCPU, dilationResultOfGPU, WIDTH, HEIGHT))
			{
				break;
			}


			logPrinter.PrintLogs("Level Discretization On CPU", Info);
			LevelDiscretizationOnCPU::LevelDiscretization(dilationResultOfCPU, WIDTH, HEIGHT, 15);

//			cudaMemcpy(originalFrameOnDeivce, dilationResultOfGPU, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			logPrinter.PrintLogs("Level Discretization On GPU", Info);
			LevelDiscretizationOnGPU(resultOnDevice, WIDTH, HEIGHT, 15);
			cudaMemcpy(dilationResultOfGPU, resultOnDevice, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

			if (!CheckDiff::Check(dilationResultOfCPU, dilationResultOfGPU, WIDTH, HEIGHT))
			{
				break;
			}

			logPrinter.PrintLogs("segementation On GPU", Info);
			Segmentation(dilationResultOfGPU, WIDTH, HEIGHT);
		}

		cudaFree(originalFrameOnDeivce);
		cudaFree(resultOnDevice);
		delete[] dilationResultOfCPU;
		delete[] dilationResultOfGPU;

		delete fileReader;
		cudaDeviceRelease();
	}

	system("Pause");
	return 0;
}
