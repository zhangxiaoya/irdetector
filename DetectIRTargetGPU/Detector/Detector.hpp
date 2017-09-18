#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <cuda_runtime_api.h>
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../Headers/GlobalMainHeaders.h"
#include "../Dilations/DilatetionKernel.cuh"
#include "../LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "../CCL/MeshCCLKernelD.cuh"

class Detector
{
public:
	explicit Detector(int _width, int _height);

	~Detector();

	bool InitSpace();

	void DetectTargets(unsigned char* frame);

private:
	void CopyFrameData(unsigned char* frame);

protected:
	bool ReleaseSpace();

private:
	int width;
	int height;

	int radius;
	int discretizationScale;

	bool isInitSpaceReady;
	bool isFrameDataReady;

	unsigned char* originalFrameOnHost;
	unsigned char* originalFrameOnDevice;
	unsigned char* dilationResultOnDevice;

	int* labelsOnDevice;
	int* referenceOfLabelsOnDevice;

	bool* modificationFlagOnDevice;
};

inline Detector::Detector(int _width = 320, int _height = 256)
	: width(_width),
	  height(_height),
	  radius(1),
	  discretizationScale(15),
	  isInitSpaceReady(true),
	  isFrameDataReady(true),
	  originalFrameOnHost(nullptr),
	  originalFrameOnDevice(nullptr),
	  dilationResultOnDevice(nullptr),
	  labelsOnDevice(nullptr),
	  referenceOfLabelsOnDevice(nullptr),
	  modificationFlagOnDevice(nullptr)
{
}

inline Detector::~Detector()
{
	ReleaseSpace();
}

inline bool Detector::ReleaseSpace()
{
	auto status = true;
	if (this->originalFrameOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->originalFrameOnHost), status);
		if (status == true)
		{
			this->originalFrameOnHost = nullptr;
		}
	}
	if (this->originalFrameOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->originalFrameOnDevice), status);
		if (status == true)
		{
			this->originalFrameOnDevice = nullptr;
		}
	}
	if (this->dilationResultOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->dilationResultOnDevice), status);
		if (status == true)
		{
			this->dilationResultOnDevice == nullptr;
		}
	}
	if(this->labelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->labelsOnDevice), status);
		if(status == true)
		{
			this->labelsOnDevice = nullptr;
		}
	}
	if(this->referenceOfLabelsOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->referenceOfLabelsOnDevice), status);
		if(status == true)
		{
			this->referenceOfLabelsOnDevice = nullptr;
		}
	}
	if(this->modificationFlagOnDevice != nullptr)
	{
		CheckCUDAReturnStatus(cudaFree(this->modificationFlagOnDevice), status);
		if (status == true)
		{
			this->modificationFlagOnDevice = nullptr;
		}
	}

	if(status == true)
	{
		logPrinter.PrintLogs("Release space success!", LogLevel::Info);
	}
	else
	{
		logPrinter.PrintLogs("Release space failed!", LogLevel::Error);
	}
	return status;
}

inline bool Detector::InitSpace()
{
	logPrinter.PrintLogs("Release space before re-init space ...", Info);
	if(ReleaseSpace() == false)
		return false;

	isInitSpaceReady = true;
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->originalFrameOnHost), sizeof(unsigned char) * width * height), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->originalFrameOnDevice), sizeof(unsigned char) * width * height),isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->dilationResultOnDevice), sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->labelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->referenceOfLabelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->modificationFlagOnDevice), sizeof(bool)),isInitSpaceReady);

	return isInitSpaceReady;
}

inline void Detector::CopyFrameData(unsigned char* frame)
{
	this->isFrameDataReady = true;

	memcpy(this->originalFrameOnHost, frame, sizeof(unsigned char) * width * height);
	CheckCUDAReturnStatus(cudaMemcpy(this->originalFrameOnDevice, this->originalFrameOnHost, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice), isFrameDataReady);

	if(isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Copy current frame data failed!", LogLevel::Error);
	}
}

inline void Detector::DetectTargets(unsigned char* frame)
{
	CopyFrameData(frame);

	if(isFrameDataReady == true)
	{
		// dilation on gpu
		DilationFilter(this->originalFrameOnDevice, this->dilationResultOnDevice, width, height, radius);

		// level disretization on gpu
		LevelDiscretizationOnGPU(this->dilationResultOnDevice, width, height, discretizationScale);

		// CCL On Device
		MeshCCL(this->dilationResultOnDevice, this->labelsOnDevice, this->referenceOfLabelsOnDevice, this->modificationFlagOnDevice, width, height);
	}
}
#endif
