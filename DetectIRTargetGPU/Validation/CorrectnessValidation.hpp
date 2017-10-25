#pragma once
#include "../DataReaderFromeFiles/BinaryFileReader.hpp"
#include "../Models/LogLevel.hpp"
#include "../LogPrinter/LogPrinter.hpp"
#include "../Dilations/DilatetionOnCPU.hpp"
#include "../Dilations/DilatetionKernel.cuh"
#include "../Checkers/CheckDiff.hpp"
#include "../LevelDiscretization/LevelDiscretizationOnCPU.hpp"
#include "../LevelDiscretization/LevelDiscretizationKernel.cuh"
#include "../Checkers/CheckCUDAReturnStatus.h"
#include "../CCL/MeshCCLOnCPU.hpp"
#include "../CCL/MeshCCLKernelD.cuh"
#include "../Checkers/CheckPerf.h"

class CorrectnessValidation
{
public:
	explicit CorrectnessValidation(const int width, const int height, BinaryFileReader* file_reader = nullptr)
		: fileReader(file_reader),
		  originalFrameOnHost(nullptr),
		  originalFrameOnDeivce(nullptr),
		  resultOfDilationOnHostUseCPU(nullptr),
		  resultOfDilationOnHostUseGPU(nullptr),
		  resultOfDilationOnDevice(nullptr),
		  resultOfLevelDiscretizationOnHostUseCPU(nullptr),
		  resultOfLevelDiscretizationOnDevice(nullptr),
		  resultOfLevelDiscretizationOnHostUseGPU(nullptr),
		  resultOfCCLOnDevice(nullptr),
		  resultOfCCLOnHostUseCPU(nullptr),
		  resultOfCCLOnHostUseGPU(nullptr),
		  referenceOfCCLOnDevice(nullptr),
		  modificationFlagOnDevice(nullptr),
		  isInitSpaceReady(false),
		  Width(width),
		  Height(height)
	{
	}

	~CorrectnessValidation()
	{
		delete fileReader;
		DestroySpace();
	}

	void InitValidationData(std::string validationFileName);

	void VailidationAll();

private:
	bool CheckFileReader() const;

	bool CheckInitSpace() const;

protected:
	bool DilationValidation() const;

	bool LevelDiscretizationValidation() const;

	bool CCLValidation() const;

	bool InitSpace();

	void DestroySpace() const;

	void ResetResultsToZero() const;

private:
	BinaryFileReader* fileReader;

	unsigned short* originalFrameOnHost;
	unsigned short* originalFrameOnDeivce;
	unsigned short* resultOfDilationOnHostUseCPU;
	unsigned short* resultOfDilationOnHostUseGPU;
	unsigned short* resultOfDilationOnDevice;
	unsigned short* resultOfLevelDiscretizationOnHostUseCPU;
	unsigned short* resultOfLevelDiscretizationOnDevice;
	unsigned short* resultOfLevelDiscretizationOnHostUseGPU;
	int* resultOfCCLOnDevice;
	int* resultOfCCLOnHostUseCPU;
	int* resultOfCCLOnHostUseGPU;
	int* referenceOfCCLOnDevice;
	bool* modificationFlagOnDevice;

	bool isInitSpaceReady;

	int Width;
	int Height;

	LogPrinter logPrinter;
};

inline void CorrectnessValidation::InitValidationData(std::string validationFileName)
{
	if(fileReader != nullptr)
	{
		delete fileReader;
		fileReader = nullptr;
	}
	fileReader = new BinaryFileReader(validationFileName);
	fileReader->ReadBinaryFileToHostMemory();

	this->InitSpace();
}

inline void CorrectnessValidation::ResetResultsToZero() const
{
	memset(resultOfDilationOnHostUseCPU, 0, Width * Height * sizeof(unsigned short));
	memset(resultOfDilationOnHostUseGPU, 0, Width * Height * sizeof(unsigned short));
	memset(resultOfLevelDiscretizationOnHostUseCPU, 0, Width * Height * sizeof(unsigned short));
	memset(resultOfLevelDiscretizationOnHostUseGPU, 0, Width * Height * sizeof(unsigned short));
	memset(resultOfCCLOnHostUseCPU, 0, Width * Height * sizeof(int));
	memset(resultOfCCLOnHostUseGPU, 0, Width * Height * sizeof(int));

	auto memcpyStatus = true;
	CheckCUDAReturnStatus(cudaMemcpy(resultOfDilationOnDevice, resultOfDilationOnHostUseGPU, sizeof(unsigned short)*Width * Height, cudaMemcpyHostToDevice), memcpyStatus);
	CheckCUDAReturnStatus(cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfLevelDiscretizationOnHostUseGPU, sizeof(unsigned short)*Width * Height, cudaMemcpyHostToDevice), memcpyStatus);
	CheckCUDAReturnStatus(cudaMemcpy(resultOfCCLOnDevice, resultOfCCLOnHostUseGPU, sizeof(int) * Width * Height, cudaMemcpyHostToDevice), memcpyStatus);
}

inline bool CorrectnessValidation::CheckFileReader() const
{
	if(fileReader == nullptr)
	{
		logPrinter.PrintLogs("File Reader is Not Ready!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline bool CorrectnessValidation::CheckInitSpace() const
{
	if (isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Init Space on Device and Host First!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline void CorrectnessValidation::VailidationAll()
{
	if (CheckFileReader()) return;
	if (CheckInitSpace()) return;

	auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the accuracy for this test file ... ", Info);
	auto checkResult = false;
	auto insideProcessSatus = true;

	char iterationText[200];

	for(unsigned i = 0;i<frameCount;++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		ResetResultsToZero();

		originalFrameOnHost = dataPoint[i];
		CheckCUDAReturnStatus(cudaMemcpy(originalFrameOnDeivce, originalFrameOnHost, sizeof(unsigned short) *Width * Height, cudaMemcpyHostToDevice), insideProcessSatus);

		// Check Dilation on GPU
		checkResult = DilationValidation();
		if(checkResult == false)
			break;

		memcpy(resultOfLevelDiscretizationOnHostUseCPU, resultOfDilationOnHostUseCPU, sizeof(unsigned short) * Width * Height);
		CheckCUDAReturnStatus(cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfDilationOnDevice, sizeof(unsigned short) * Width * Height, cudaMemcpyDeviceToDevice), insideProcessSatus);

		// Check Discretization on GPU
		checkResult = LevelDiscretizationValidation();
		if(checkResult == false)
			break;

		// Check CCL On GPU
		checkResult = CCLValidation();
		if (checkResult == false)
			break;
	}
	if(checkResult == true)
	{
		logPrinter.PrintLogs("All test cases passed!", Info);
	}
	else
	{
		logPrinter.PrintLogs("One or many test cases failed!", Waring);
	}
}

inline bool CorrectnessValidation::DilationValidation() const
{
	auto dilationRadius = 1;

	logPrinter.PrintLogs("Dilation on CPU!", Info);
	DilationOnCPU::DilationCPU(originalFrameOnHost, resultOfDilationOnHostUseCPU, Width, Height, dilationRadius);

	auto insideStatus = true;
	logPrinter.PrintLogs("Dialtion On GPU", Info);
	CheckPerf(NaiveDilation(originalFrameOnDeivce, resultOfDilationOnDevice, Width, Height, dilationRadius), "Naive Dilation On GPU");
	CheckCUDAReturnStatus(cudaMemcpy(resultOfDilationOnHostUseGPU, resultOfDilationOnDevice, Width * Height * sizeof(unsigned short), cudaMemcpyDeviceToHost), insideStatus);

	return CheckDiff::Check<unsigned short>(resultOfDilationOnHostUseCPU, resultOfDilationOnHostUseGPU, Width, Height);
}

inline bool CorrectnessValidation::LevelDiscretizationValidation() const
{
	auto discretization_scale = 15;

	logPrinter.PrintLogs("Level Discretization On CPU", Info);
	LevelDiscretizationOnCPU::LevelDiscretization(resultOfLevelDiscretizationOnHostUseCPU, Width, Height, discretization_scale);

	auto insideStatus = true;
	logPrinter.PrintLogs("Level Discretization On GPU", Info);
	CheckPerf(LevelDiscretizationOnGPU(resultOfLevelDiscretizationOnDevice, Width, Height, discretization_scale), "Discretization On GPU");
	CheckCUDAReturnStatus(cudaMemcpy(resultOfLevelDiscretizationOnHostUseGPU, resultOfLevelDiscretizationOnDevice, sizeof(unsigned short) * Width * Height, cudaMemcpyDeviceToHost), insideStatus);

	return CheckDiff::Check<unsigned short>(resultOfLevelDiscretizationOnHostUseCPU, resultOfLevelDiscretizationOnHostUseGPU, Width, Height);
}

inline bool CorrectnessValidation::CCLValidation() const
{
	logPrinter.PrintLogs("CCL On CPU", Info);
	MeshCCLOnCPU::ccl(resultOfLevelDiscretizationOnHostUseCPU, resultOfCCLOnHostUseCPU, Width, Height, 4, 0);

	auto insideStatus = true;
	logPrinter.PrintLogs("CCL On GPU", Info);
	CheckPerf(MeshCCL(resultOfLevelDiscretizationOnHostUseGPU, resultOfCCLOnDevice, referenceOfCCLOnDevice, modificationFlagOnDevice, Width, Height), "CCL On GPU");
	CheckCUDAReturnStatus(cudaMemcpy(resultOfCCLOnHostUseGPU, resultOfCCLOnDevice, sizeof(int) * Width * Height, cudaMemcpyDeviceToHost), insideStatus);
	return CheckDiff::Check<int>(resultOfCCLOnHostUseCPU, resultOfCCLOnHostUseGPU, Width, Height);
}

inline bool CorrectnessValidation::InitSpace()
{
	isInitSpaceReady = true;

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->originalFrameOnDeivce), sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->resultOfDilationOnDevice), sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->resultOfLevelDiscretizationOnDevice), sizeof(unsigned short) * Height * Width), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->resultOfCCLOnDevice), sizeof(int) * Height * Width), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->referenceOfCCLOnDevice), sizeof(int) * Height * Width), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc(reinterpret_cast<void**>(&this->modificationFlagOnDevice), sizeof(bool)), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfDilationOnHostUseCPU),sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfDilationOnHostUseGPU),sizeof(unsigned short) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfLevelDiscretizationOnHostUseCPU), sizeof(unsigned short) *Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfLevelDiscretizationOnHostUseGPU), sizeof(unsigned short) *Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfCCLOnHostUseCPU), sizeof(int) * Width * Height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost(reinterpret_cast<void**>(&this->resultOfCCLOnHostUseGPU), sizeof(int) * Width * Height), isInitSpaceReady);

	return isInitSpaceReady;
}

inline void CorrectnessValidation::DestroySpace() const
{
	bool status;

	CheckCUDAReturnStatus(cudaFree(this->originalFrameOnDeivce), status);
	CheckCUDAReturnStatus(cudaFree(this->resultOfDilationOnDevice), status);
	CheckCUDAReturnStatus(cudaFree(this->resultOfLevelDiscretizationOnDevice), status);
	CheckCUDAReturnStatus(cudaFree(this->resultOfCCLOnDevice), status);
	CheckCUDAReturnStatus(cudaFree(this->referenceOfCCLOnDevice), status);

	CheckCUDAReturnStatus(cudaFree(this->modificationFlagOnDevice), status);

	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfDilationOnHostUseCPU), status);
	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfDilationOnHostUseGPU), status);
	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfLevelDiscretizationOnHostUseCPU),  status);
	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfLevelDiscretizationOnHostUseGPU),  status);
	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfCCLOnHostUseCPU), status);
	CheckCUDAReturnStatus(cudaFreeHost(this->resultOfCCLOnHostUseGPU), status);
}
