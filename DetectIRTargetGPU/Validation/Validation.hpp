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

class Validation
{
public:
	explicit Validation(BinaryFileReader* file_reader = nullptr)
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
		  width(320),
		  height(256)
	{
	}

	~Validation()
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

	void InitSpace();

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

	int width;
	int height;

	LogPrinter logPrinter;
};

inline void Validation::InitValidationData(std::string validationFileName)
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

inline void Validation::ResetResultsToZero() const
{
	memset(resultOfDilationOnHostUseCPU, 0, width * height * sizeof(unsigned char));
	memset(resultOfDilationOnHostUseGPU, 0, width * height * sizeof(unsigned char));
	memset(resultOfLevelDiscretizationOnHostUseCPU, 0, width * height * sizeof(unsigned char));
	memset(resultOfLevelDiscretizationOnHostUseGPU, 0, width * height * sizeof(unsigned char));
	memset(resultOfCCLOnHostUseCPU, 0, width * height * sizeof(int));
	memset(resultOfCCLOnHostUseGPU, 0, width * height * sizeof(int));

	cudaMemcpy(resultOfDilationOnDevice, resultOfDilationOnHostUseGPU, sizeof(unsigned char)*width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfLevelDiscretizationOnHostUseGPU, sizeof(unsigned char)*width * height, cudaMemcpyHostToDevice);
	cudaMemcpy(resultOfCCLOnDevice, resultOfCCLOnHostUseGPU, sizeof(int) * width * height, cudaMemcpyHostToDevice);
}

inline bool Validation::CheckFileReader() const
{
	if(fileReader == nullptr)
	{
		logPrinter.PrintLogs("File Reader is Not Ready!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline bool Validation::CheckInitSpace() const
{
	if (isInitSpaceReady == false)
	{
		logPrinter.PrintLogs("Init Space on Device and Host First!", Error);
		system("Pause");
		return true;
	}
	return false;
}

inline void Validation::VailidationAll()
{
	if (CheckFileReader()) return;
	if (CheckInitSpace()) return;

	auto frameCount = fileReader->GetFrameCount();
	auto dataPoint = fileReader->GetDataPoint();

	logPrinter.PrintLogs("Test the accuracy for this test file ... ", Info);
	auto checkResult = false;
	char iterationText[200];

	for(auto i = 0;i<frameCount;++i)
	{
		sprintf_s(iterationText, 200, "Checking for frame %04d ...", i);
		logPrinter.PrintLogs(iterationText, Info);

		ResetResultsToZero();

		originalFrameOnHost = dataPoint[i];
		cudaMemcpy(originalFrameOnDeivce, originalFrameOnHost, sizeof(unsigned char) *width * height, cudaMemcpyHostToDevice);

		checkResult = DilationValidation();
		if(checkResult == false)
			break;

		memcpy(resultOfLevelDiscretizationOnHostUseCPU, resultOfDilationOnHostUseCPU, sizeof(unsigned char) * width * height);
		cudaMemcpy(resultOfLevelDiscretizationOnDevice, resultOfDilationOnDevice, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToDevice);

		checkResult = LevelDiscretizationValidation();
		if(checkResult == false)
			break;

		checkResult = CCLValidation();
		if (checkResult == false)
			break;
	}
	if(checkResult == true)
	{
		logPrinter.PrintLogs("All test cases passed!", Info);
	}
}

inline bool Validation::DilationValidation() const
{
	logPrinter.PrintLogs("Dilation on CPU!", Info);
	DilationOnCPU::DilationCPU(originalFrameOnHost, resultOfDilationOnHostUseCPU, width, height, 1);

	logPrinter.PrintLogs("Dialtion On GPU", Info);
	CheckPerf(DilationFilter(originalFrameOnDeivce, resultOfDilationOnDevice, width, height, 1), "Dilation on GPU");
	cudaMemcpy(resultOfDilationOnHostUseGPU, resultOfDilationOnDevice, width * height, cudaMemcpyDeviceToHost);

	return CheckDiff::Check<unsigned short>(resultOfDilationOnHostUseCPU, resultOfDilationOnHostUseGPU, width, height);
}

inline bool Validation::LevelDiscretizationValidation() const
{
	logPrinter.PrintLogs("Level Discretization On CPU", Info);
	LevelDiscretizationOnCPU::LevelDiscretization(resultOfLevelDiscretizationOnHostUseCPU, width, height, 15);

	logPrinter.PrintLogs("Level Discretization On GPU", Info);
	CheckPerf(LevelDiscretizationOnGPU(resultOfLevelDiscretizationOnDevice, width, height, 15),"Discretization On GPU");
	cudaMemcpy(resultOfLevelDiscretizationOnHostUseGPU, resultOfLevelDiscretizationOnDevice, width * height, cudaMemcpyDeviceToHost);

	return CheckDiff::Check<unsigned short>(resultOfLevelDiscretizationOnHostUseCPU, resultOfLevelDiscretizationOnHostUseGPU, width, height);
}

inline bool Validation::CCLValidation() const
{
	logPrinter.PrintLogs("CCL On CPU", Info);
	MeshCCLOnCPU::ccl(resultOfLevelDiscretizationOnHostUseCPU, resultOfCCLOnHostUseCPU, width, height, 4, 0);

	logPrinter.PrintLogs("CCL On GPU", Info);
	CheckPerf(MeshCCL(resultOfLevelDiscretizationOnHostUseGPU, resultOfCCLOnDevice, referenceOfCCLOnDevice, modificationFlagOnDevice, width, height), "CCL On GPU");
	cudaMemcpy(resultOfCCLOnHostUseGPU, resultOfCCLOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	return CheckDiff::Check<int>(resultOfCCLOnHostUseCPU, resultOfCCLOnHostUseGPU, width, height);
}

inline void Validation::InitSpace()
{
	isInitSpaceReady = true;

	CheckCUDAReturnStatus(cudaMalloc((void**)&this->originalFrameOnDeivce, sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->resultOfDilationOnDevice, sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->resultOfLevelDiscretizationOnDevice, sizeof(unsigned char) * height * width), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->resultOfCCLOnDevice, sizeof(int) * height * width), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMalloc((void**)&this->referenceOfCCLOnDevice, sizeof(int) * height * width), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMalloc((void**)&this->modificationFlagOnDevice, sizeof(bool)), isInitSpaceReady);

	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfDilationOnHostUseCPU,sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfDilationOnHostUseGPU,sizeof(unsigned char) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfLevelDiscretizationOnHostUseCPU, sizeof(unsigned char) *width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfLevelDiscretizationOnHostUseGPU, sizeof(unsigned char) *width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfCCLOnHostUseCPU, sizeof(int) * width * height), isInitSpaceReady);
	CheckCUDAReturnStatus(cudaMallocHost((void**)&this->resultOfCCLOnHostUseGPU, sizeof(int) * width * height), isInitSpaceReady);
}

inline void Validation::DestroySpace() const
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
