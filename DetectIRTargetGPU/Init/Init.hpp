#pragma once
#include "../Models/LogLevel.hpp"
#include "../Headers/GlobalMainHeaders.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include "../Checkers/CheckCUDAReturnStatus.h"

class CUDAInit
{
public:
	static bool cudaDeviceInit();

	static void cudaDeviceRelease();
};

inline bool CUDAInit::cudaDeviceInit()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		logPrinter.PrintLogs("CUDA error: no devices supporting CUDA.", Error);
		return false;
	}

	auto status = true;
	CheckCUDAReturnStatus(cudaSetDevice(0), status);

	if (status == false)
	{
		logPrinter.PrintLogs("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", Error);
		return status;
	}

	logPrinter.PrintLogs("cudaSetDevice success!", Info);
	return status;
}

inline void CUDAInit::cudaDeviceRelease()
{
	auto status = true;
	CheckCUDAReturnStatus(cudaDeviceReset(), status);

	if (status == true)
	{
		logPrinter.PrintLogs("cudaDeviceReset success!", Info);
	}
}
