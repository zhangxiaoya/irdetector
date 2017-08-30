#include <iostream>

#include "Headers/GlobalMainHeaders.h"

inline bool cudaDeviceInit(int argc, const char** argv)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		logPrinter.PrintLogs("CUDA error: no devices supporting CUDA.", LogLevel::Error);
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	auto cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		logPrinter.PrintLogs("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", LogLevel::Error);
		return false;
	}
	logPrinter.PrintLogs("cudaSetDevice success!", LogLevel::Info);
	return true;
}

void cudaDeviceRelease()
{
	auto cudaResetStatus = cudaDeviceReset();
	if(cudaResetStatus == cudaSuccess)
	{
		logPrinter.PrintLogs("cudaDeviceReset success!", LogLevel::Info);
	}
	else
	{
		logPrinter.PrintLogs("cudaDeviceReset failed!", LogLevel::Waring);
	}
}

int main(int argc, char* argv[])
{
	auto cudaInitStatus = cudaDeviceInit(argc, const_cast<const char **>(argv));
	if(cudaInitStatus)
	{
		cudaDeviceRelease();
	}

	system("Pause");
	return 0;
}
