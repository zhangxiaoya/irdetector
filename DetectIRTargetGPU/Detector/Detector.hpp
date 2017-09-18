#ifndef __DETECTOR_H__
#define __DETECTOR_H__
#include <cuda_runtime_api.h>
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../Headers/GlobalMainHeaders.h"
#include "../Dilations/DilatetionKernel.cuh"
#include "../LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "../CCL/MeshCCLKernelD.cuh"
#include "../Models/FourLimits.h"

class Detector
{
public:
	explicit Detector(int _width, int _height);

	~Detector();

	bool InitSpace();

	void DetectTargets(unsigned char* frame);

private:
	void CopyFrameData(unsigned char* frame);

	static void GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height);

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

	int* labelsOnHost;
	int* labelsOnDevice;
	int* referenceOfLabelsOnDevice;

	bool* modificationFlagOnDevice;

	FourLimits* allObjects;
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
	  labelsOnHost(nullptr),
	  labelsOnDevice(nullptr),
	  referenceOfLabelsOnDevice(nullptr),
	  modificationFlagOnDevice(nullptr),
	  allObjects(nullptr)
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
	if(this->labelsOnHost != nullptr)
	{
		CheckCUDAReturnStatus(cudaFreeHost(this->labelsOnHost), status);
		if(status == true)
		{
			this->labelsOnHost = nullptr;
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

	allObjects = static_cast<FourLimits*>(malloc(sizeof(FourLimits) * width * height));

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
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->labelsOnHost), sizeof(int) * width * height), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->originalFrameOnDevice), sizeof(unsigned char) * width * height),isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->dilationResultOnDevice), sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->labelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->referenceOfLabelsOnDevice), sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->modificationFlagOnDevice), sizeof(bool)),isInitSpaceReady);

	delete[] allObjects;
	return isInitSpaceReady;
}

inline void Detector::CopyFrameData(unsigned char* frame)
{
	this->isFrameDataReady = true;

	memcpy(this->originalFrameOnHost, frame, sizeof(unsigned char) * width * height);
	memset(this->allObjects, -1, sizeof(FourLimits) * width * height);

	CheckCUDAReturnStatus(cudaMemcpy(this->originalFrameOnDevice, this->originalFrameOnHost, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice), isFrameDataReady);
	if(isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Copy current frame data failed!", LogLevel::Error);
	}
}

inline void Detector::GetAllObjects(int* labelsOnHost, FourLimits* allObjects, int width, int height)
{
	// top
	for (auto r = 0; r < height; ++r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].top == -1)
				allObjects[label].top = r;
		}
	}
	// bottom
	for (auto r = height - 1; r >= 0; --r)
	{
		for (auto c = 0; c < width; ++c)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].bottom == -1)
				allObjects[label].bottom = r;
		}
	}

	// left
	for (auto c = 0; c < width; ++c)
	{
		for (auto r = 0; r < height; ++r)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].left == -1)
				allObjects[label].left = c;
		}
	}
	// right
	for (auto c = width - 1; c >= 0; --c)
	{
		for (auto r = 0; r < height; ++r)
		{
			auto label = labelsOnHost[r * width + c];
			if (allObjects[label].right == -1)
				allObjects[label].right = c;
		}
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

		// copy labels from device to host
		cudaMemcpy(this->labelsOnHost, this->labelsOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);

		// get all object
		GetAllObjects(labelsOnHost, allObjects, width, height);
	}
}
#endif
