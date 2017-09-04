#include <iostream>

#include "Headers/GlobalMainHeaders.h"
#include "DataReaderFromeFiles/BinaryFileReader.hpp"
#include "Dilations/DilatetionOnCPU.hpp"
#include "Dilations/DilatetionKernel.h"
#include "Checkers/CheckDiff.hpp"
#include "LevelDiscretization/LevelDiscretizationOnCPU.hpp"
#include "LevelDiscretization/LevelDiscretizationKernel.cuh"

inline bool cudaDeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		logPrinter.PrintLogs("CUDA error: no devices supporting CUDA.", Error);
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
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
		unsigned char* frameOnDeivce;
		unsigned char* resultOnDevice;

		cudaMalloc(&frameOnDeivce, WIDTH * HEIGHT);
		cudaMalloc(&resultOnDevice, WIDTH * HEIGHT);

		auto dilationResultOnCPU = new unsigned char[WIDTH * HEIGHT];
		auto dilationResultOnGPU = new unsigned char[WIDTH * HEIGHT];

		std::string fileName = "C:\\D\\Cabins\\Projects\\Project1\\binaryFiles\\ir_file_20170531_1000m_1_partOne.bin";

		auto fileReader = new BinaryFileReader(fileName);

		fileReader->ReadBinaryFileToHostMemory();
		auto frameCount = fileReader->GetFrameCount();
		auto dataPoint = fileReader->GetDataPoint();

		auto iterationText = new char[200];
		for (auto i = 0; i < frameCount; ++i)
		{
			memset(dilationResultOnGPU, 0, WIDTH * HEIGHT);
			memset(dilationResultOnCPU, 0, WIDTH * HEIGHT);

			auto perFrame = dataPoint[i];

			logPrinter.PrintLogs("Dilation on CPU!", Info);
			DilationOnCPU::dilationCPU(perFrame, dilationResultOnCPU, WIDTH, HEIGHT, 1);

			sprintf_s(iterationText, 200, "Copy the %04d frame to device", i);
			logPrinter.PrintLogs(iterationText, Info);
			cudaMemcpy(frameOnDeivce, perFrame, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

			logPrinter.PrintLogs("Dialtion On GPU", Info);
			FilterDilation(frameOnDeivce, resultOnDevice, WIDTH, HEIGHT, 1);

			cudaMemcpy(dilationResultOnGPU, resultOnDevice, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

			if (!CheckDiff::Check(dilationResultOnCPU, dilationResultOnGPU, WIDTH, HEIGHT))
			{
				break;
			}


			logPrinter.PrintLogs("Level Discretization On CPU", Info);
			LevelDiscretizationOnCPU::LevelDiscretization(dilationResultOnCPU, WIDTH, HEIGHT, 15);

			cudaMemcpy(frameOnDeivce, dilationResultOnGPU, WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			logPrinter.PrintLogs("Level Discretization On GPU", Info);
			LevelDiscretizationOnGPU(frameOnDeivce, WIDTH, HEIGHT, 15);
			cudaMemcpy(dilationResultOnGPU, frameOnDeivce, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

			if (!CheckDiff::Check(dilationResultOnCPU, dilationResultOnGPU, WIDTH, HEIGHT))
			{
				break;
			}
		}

		cudaFree(frameOnDeivce);
		cudaFree(resultOnDevice);
		delete[] dilationResultOnCPU;
		delete[] dilationResultOnGPU;

		delete fileReader;
		cudaDeviceRelease();
	}

	system("Pause");
	return 0;
}
